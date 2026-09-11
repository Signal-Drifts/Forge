import assert from 'node:assert/strict';
import test from 'node:test';

import { backend } from '../lib/backend.ts';

void test('invalid backend error bodies produce a usable message', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () =>
    new Response('<html>bad gateway</html>', { status: 502 });
  try {
    await assert.rejects(
      backend('/health'),
      /The request could not be completed\. Please try later\./,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

void test('invalid successful backend bodies are rejected', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response('', { status: 200 });
  try {
    await assert.rejects(
      backend('/health'),
      /The backend returned an invalid response\./,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});
