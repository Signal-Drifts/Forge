import assert from 'node:assert/strict';
import test from 'node:test';

import {
  getOptimizationPrompt,
  precisionOptions,
} from '../lib/optimization-rules.ts';

const base = {
  prune: false,
  method: '',
  ratio: '',
  hasDataset: false,
  quantize: true,
};

for (const sourceBits of precisionOptions) {
  for (const targetBits of precisionOptions) {
    void test(`${sourceBits}-bit model to ${targetBits}-bit target`, () => {
      const prompt = getOptimizationPrompt({
        ...base,
        sourceBits,
        targetBits: String(targetBits),
      });
      if (sourceBits === targetBits) {
        assert.equal(prompt.canRun, false);
        assert.match(prompt.items[0].message, /different/);
      } else {
        assert.equal(prompt.canRun, true);
        if (targetBits < sourceBits) {
          assert.equal(prompt.items[0].title, 'Quantize');
          assert.match(prompt.items[0].message, /may reduce precision/);
        } else {
          assert.deepEqual(prompt.items, []);
        }
      }
    });
  }
}

for (const sourceBits of [4, 8]) {
  void test(`${sourceBits}-bit model cannot be pruned`, () => {
    const prompt = getOptimizationPrompt({
      ...base,
      sourceBits,
      prune: true,
      method: 'magnitude',
      ratio: '10',
      quantize: false,
      targetBits: '',
    });
    assert.equal(prompt.canRun, false);
    assert.equal(
      prompt.items[0].message,
      'Pruning requires a 16-bit, 24-bit, or 32-bit model.',
    );
  });
}

for (const sourceBits of [16, 24, 32]) {
  void test(`${sourceBits}-bit model supports pruning`, () => {
    const prompt = getOptimizationPrompt({
      ...base,
      sourceBits,
      prune: true,
      method: 'magnitude',
      ratio: '10',
      quantize: false,
      targetBits: '',
    });
    assert.equal(prompt.canRun, true);
    assert.equal(prompt.items[0].title, 'Prune');
  });
}

void test('pruning validates method and percentage boundaries', () => {
  const selection = {
    ...base,
    sourceBits: 16,
    prune: true,
    quantize: false,
    targetBits: '',
  };
  assert.match(getOptimizationPrompt(selection).items[0].message, /method/);
  for (const ratio of ['', '0', '81']) {
    assert.equal(
      getOptimizationPrompt({ ...selection, method: 'magnitude', ratio })
        .canRun,
      false,
    );
  }
  for (const ratio of ['1', '80']) {
    assert.equal(
      getOptimizationPrompt({ ...selection, method: 'magnitude', ratio })
        .canRun,
      true,
    );
  }
});

void test('N:M does not require a percentage', () => {
  const prompt = getOptimizationPrompt({
    ...base,
    sourceBits: 16,
    prune: true,
    method: 'nm',
    quantize: false,
    targetBits: '',
  });
  assert.equal(prompt.canRun, true);
});

for (const method of ['wanda', 'sparsegpt']) {
  void test(`${method} requires a calibration dataset`, () => {
    const selection = {
      ...base,
      sourceBits: 16,
      prune: true,
      method,
      ratio: '10',
      quantize: false,
      targetBits: '',
    };
    assert.equal(getOptimizationPrompt(selection).canRun, false);
    assert.match(getOptimizationPrompt(selection).items[0].message, /dataset/);
    assert.equal(
      getOptimizationPrompt({ ...selection, hasDataset: true }).canRun,
      true,
    );
  });
}

void test('combined confirmation follows backend order', () => {
  const prompt = getOptimizationPrompt({
    ...base,
    sourceBits: 16,
    prune: true,
    method: 'magnitude',
    ratio: '10',
    quantize: true,
    targetBits: '4',
  });
  assert.equal(prompt.canRun, true);
  assert.deepEqual(
    prompt.items.map((item) => item.title),
    ['Prune', 'Quantize'],
  );
});

for (const sourceBits of [16, 24, 32]) {
  for (const targetBits of precisionOptions.filter(
    (value) => value !== sourceBits,
  )) {
    void test(`${sourceBits}-bit combined prune and ${targetBits}-bit quantize is valid`, () => {
      const prompt = getOptimizationPrompt({
        ...base,
        sourceBits,
        prune: true,
        method: 'magnitude',
        ratio: '10',
        quantize: true,
        targetBits: String(targetBits),
      });
      assert.equal(prompt.canRun, true);
      assert.deepEqual(
        prompt.items.map((item) => item.title),
        targetBits < sourceBits ? ['Prune', 'Quantize'] : ['Prune'],
      );
    });
  }
}

for (const sourceBits of [4, 8]) {
  void test(`${sourceBits}-bit combined run is blocked by pruning`, () => {
    const prompt = getOptimizationPrompt({
      ...base,
      sourceBits,
      prune: true,
      method: 'magnitude',
      ratio: '10',
      quantize: true,
      targetBits: sourceBits === 4 ? '8' : '4',
    });
    assert.equal(prompt.canRun, false);
    assert.deepEqual(
      prompt.items.map((item) => item.title),
      ['Prune'],
    );
  });
}

void test('missing operations and target precision are reported', () => {
  assert.equal(
    getOptimizationPrompt({
      ...base,
      sourceBits: 16,
      quantize: false,
      targetBits: '',
    }).canRun,
    false,
  );
  assert.equal(
    getOptimizationPrompt({ ...base, sourceBits: 16, targetBits: '' }).canRun,
    false,
  );
});

void test('unchecked pruning fields are ignored', () => {
  const prompt = getOptimizationPrompt({
    ...base,
    sourceBits: 16,
    prune: false,
    method: '',
    ratio: '',
    quantize: true,
    targetBits: '4',
  });
  assert.equal(prompt.canRun, true);
  assert.deepEqual(
    prompt.items.map((item) => item.title),
    ['Quantize'],
  );
});

void test('unchecked quantization fields are ignored', () => {
  const prompt = getOptimizationPrompt({
    ...base,
    sourceBits: 16,
    prune: true,
    method: 'magnitude',
    ratio: '10',
    quantize: false,
    targetBits: '',
  });
  assert.equal(prompt.canRun, true);
  assert.deepEqual(
    prompt.items.map((item) => item.title),
    ['Prune'],
  );
});
