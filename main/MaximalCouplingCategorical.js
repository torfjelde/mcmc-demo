"use strict";

const zip = (a, b) => a.map((k, i) => [k, b[i]]);
const sum = x => x.reduce((a, b) => a + b, 0);
const total_variation_dist = (a, b) => sum(zip(a, b).map(([x, y]) => Math.abs(x - y)));

class MaximalCouplingCategorical {
  constructor(probs_1, probs_2) {
    this.probs_1 = probs_1;
    this.probs_2 = probs_2;
    this.dim = probs_1.length;

    this.probs_min = zip(probs_1, probs_2).map(([p1, p2]) => Math.min(p1, p2));
    this.Z = sum(this.probs_min);
    this.tv_dist = total_variation_dist(probs_1, probs_2);
  }

  getSample() {
    const u = Math.random();

    const omega = 1 - this.tv_dist;
    if (u > omega) {
      let denom = 1 - this.Z;
      let dist_1 = new Categorical(this.probs_1.map((p, i) => (p - this.probs_min[i]) / denom));
      let dist_2 = new Categorical(this.probs_2.map((p, i) => (p - this.probs_min[i]) / denom));
      return [dist_1.getSample(), dist_2.getSample()];
    } else {
      let dist = new Categorical(this.probs_min.map(p => p / this.Z));
      let i = dist.getSample();
      return [i, i];
    }
  }

  static getSample(probs_1, probs_2) {
    const dist = new MaximalCouplingCategorical(probs_1, probs_2);
    return dist.getSample();
  }
}

