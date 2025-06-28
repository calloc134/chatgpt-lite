import { createParser, ParsedEvent, ReconnectInterval } from 'eventsource-parser'
import { NextRequest, NextResponse } from 'next/server'
import OpenAI from 'openai'

export const runtime = 'edge'

export interface Message {
  role: string
  content: string
}

const openaiClient = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
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
