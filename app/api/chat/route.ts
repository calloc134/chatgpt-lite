import { createParser, ParsedEvent, ReconnectInterval } from 'eventsource-parser'
import { NextRequest, NextResponse } from 'next/server'
import OpenAI from 'openai'

export const runtime = 'edge'

export interface Message {
  role: string
  content: string
}

const openaiClient = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_API_BASE_URL // 未設定なら自動で api.openai.com
})

export async function POST(req: NextRequest) {
  try {
    const { prompt, messages, input } = (await req.json()) as {
      prompt: string
      messages: Message[]
      input: string
    }
    const messagesWithHistory = [
      { content: prompt, role: 'system' },
      ...messages,
      { content: input, role: 'user' }
    ]

    // const { apiUrl, apiKey, model } = getApiConfig()
    const model = process.env.OPENAI_MODEL || 'o4-mini'
    const stream = await getResponseStream(model, messagesWithHistory)

    // const stream = await getOpenAIStream(apiUrl, apiKey, model, messagesWithHistory)
    return new NextResponse(stream, {
      headers: { 'Content-Type': 'text/event-stream' }
    })
  } catch (error) {
    console.error(error)
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    )
  }
}

const getResponseStream = async (model: string, messages: Message[]) => {
  const encoder = new TextEncoder()

  // ① system プロンプトは instructions、残り会話は 1 本の文字列として渡す
  const [{ content: instructions }, ...rest] = messages
  const userConversation = rest.map((m) => m.content).join('\n')

  const aiStream = await openaiClient.responses.create({
    model,
    instructions,
    input: userConversation,
    stream: true,
    temperature: 0.5,
    top_p: 0.95,
    // presence_penalty: 0,
    // frequency_penalty: 0,
    tools: [{ type: 'web_search_preview' }]
  })

  return new ReadableStream({
    async start(controller) {
      try {
        for await (const event of aiStream) {
          if (event.type === 'response.output_text.delta') {
            controller.enqueue(encoder.encode(event.delta))
          } else if (event.type === 'response.output_text.done') {
            controller.close()
          }
        }
      } catch (err) {
        controller.error(err)
      }
    }
  })
}

const getApiConfig = () => {
  const useAzureOpenAI =
    process.env.AZURE_OPENAI_API_BASE_URL && process.env.AZURE_OPENAI_API_BASE_URL.length > 0

  let apiUrl: string
  let apiKey: string
  let model: string
  if (useAzureOpenAI) {
    let apiBaseUrl = process.env.AZURE_OPENAI_API_BASE_URL
    const apiVersion = '2024-02-01'
    const deployment = process.env.AZURE_OPENAI_DEPLOYMENT || ''
    if (apiBaseUrl && apiBaseUrl.endsWith('/')) {
      apiBaseUrl = apiBaseUrl.slice(0, -1)
    }
    apiUrl = `${apiBaseUrl}/openai/deployments/${deployment}/chat/completions?api-version=${apiVersion}`
    apiKey = process.env.AZURE_OPENAI_API_KEY || ''
    model = '' // Azure Open AI always ignores the model and decides based on the deployment name passed through.
  } else {
    let apiBaseUrl = process.env.OPENAI_API_BASE_URL || 'https://api.openai.com'
    if (apiBaseUrl && apiBaseUrl.endsWith('/')) {
      apiBaseUrl = apiBaseUrl.slice(0, -1)
    }
    apiUrl = `${apiBaseUrl}/v1/chat/completions`
    apiKey = process.env.OPENAI_API_KEY || ''
    model = process.env.OPENAI_MODEL || 'gpt-3.5-turbo'
  }

  return { apiUrl, apiKey, model }
}

const getOpenAIStream = async (
  apiUrl: string,
  apiKey: string,
  model: string,
  messages: Message[]
) => {
  const encoder = new TextEncoder()
  const decoder = new TextDecoder()
  const res = await fetch(apiUrl, {
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
      'api-key': `${apiKey}`
    },
    method: 'POST',
    body: JSON.stringify({
      model: model,
      frequency_penalty: 0,
      max_tokens: 4000,
      messages: messages,
      presence_penalty: 0,
      stream: true,
      temperature: 0.5,
      top_p: 0.95
    })
  })

  if (res.status !== 200) {
    const statusText = res.statusText
    const responseBody = await res.text()
    console.error(`OpenAI API response error: ${responseBody}`)
    throw new Error(
      `The OpenAI API has encountered an error with a status code of ${res.status} ${statusText}: ${responseBody}`
    )
  }

  return new ReadableStream({
    async start(controller) {
      const onParse = (event: ParsedEvent | ReconnectInterval) => {
        if (event.type === 'event') {
          const data = event.data

          if (data === '[DONE]') {
            controller.close()
            return
          }

          try {
            const json = JSON.parse(data)
            const text = json.choices[0]?.delta?.content
            if (text !== undefined) {
              const queue = encoder.encode(text)
              controller.enqueue(queue)
            } else {
              console.error('Received undefined content:', json)
            }
          } catch (e) {
            console.error('Error parsing event data:', e)
            controller.error(e)
          }
        }
      }

      const parser = createParser(onParse)

      for await (const chunk of res.body as any) {
        // An extra newline is required to make AzureOpenAI work.
        const str = decoder.decode(chunk).replace('[DONE]\n', '[DONE]\n\n')
        parser.feed(str)
      }
    }
  })
}
