import { afterEach, describe, expect, it, vi } from 'vitest'
import { uploadChat } from './client'

afterEach(() => vi.restoreAllMocks())

describe('uploadChat', () => {
  it('posts file and returns summary', async () => {
    const summary = { upload_id: 'u1', users: ['a', 'b'], message_count: 2, first_date: '2025-01-01', last_date: '2025-02-01' }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(JSON.stringify(summary), { status: 200 }),
    ))
    const file = new File(['Date,User,Message'], 'chat.csv', { type: 'text/csv' })
    await expect(uploadChat(file)).resolves.toEqual(summary)
  })

  it('throws ApiError with server detail on 400', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: '분석 가능한 메시지가 없습니다.' }), { status: 400 }),
    ))
    const file = new File([''], 'bad.csv', { type: 'text/csv' })
    await expect(uploadChat(file)).rejects.toThrow('분석 가능한 메시지가 없습니다.')
  })
})
