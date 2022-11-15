"use strict";

function softmax(xs) {
  const max = Math.max(...xs);
  const tmp = xs.map(x => x - max).map(Math.exp)
  const s = tmp.reduce((a, b) => a + b);
  return tmp.map(x => x / s);
}

class Categorical {
  constructor(probs) {
    this.probs = probs;
    this.dim = probs.length;
  }

  getSample() {
    const u = Math.random();
    let sum = 0;
    for (let i = 0; i < this.dim; i++) {
      sum += this.probs[i];
      if (u < sum) {
        return i;
      }
    }

    // Return the last one if none has been accepted yet.
    return this.dim - 1;
  }

  static getSample(probs) {
    const dist = new Categorical(probs);
    return dist.getSample();
  }
}
