import assert from 'node:assert/strict';
import test from 'node:test';
import { getRunSignature } from '../lib/optimization-run.ts';

const selection = {
  modelId: 'model-1',
  modelPath: '/models/model-1',
  prune: true,
  method: 'magnitude',
  ratio: '20',
  quantize: true,
  bits: '8',
  fineTune: false,
  fineTuneMethod: '',
  iterations: '',
  datasetId: '',
};

void test('unchanged model and settings produce the same run signature', () => {
  assert.equal(getRunSignature(selection), getRunSignature({ ...selection }));
});

void test('changing any optimization value produces a different run signature', () => {
  const original = getRunSignature(selection);
  for (const changed of [
    { ...selection, modelId: 'model-2' },
    { ...selection, prune: false },
    { ...selection, ratio: '30' },
    { ...selection, bits: '4' },
    { ...selection, fineTuneMethod: 'qlora' },
  ]) {
    assert.notEqual(getRunSignature(changed), original);
  }
});
