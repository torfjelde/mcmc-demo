"use strict";

MCMC.registerAlgorithm("MultinomialHMC", {
  description: "Hamiltonian Monte Carlo",

  about: () => {
    window.open("https://en.wikipedia.org/wiki/Hybrid_Monte_Carlo");
  },

  init: (self) => {
    self.leapfrogSteps = 37;
    self.dt = 0.1;
  },

  reset: (self) => {
    self.chain = [MultivariateNormal.getSample(self.dim)];
  },

  attachUI: (self, folder) => {
    folder.add(self, "leapfrogSteps", 5, 100).step(1).name("Leapfrog Steps");
    folder.add(self, "dt", 0.05, 0.5).step(0.025).name("Leapfrog &Delta;t");
    folder.open();
  },

  step: (self, visualizer) => {
    const q0 = self.chain.last();
    const p0 = MultivariateNormal.getSample(self.dim);

    // sample index of initial point uniformly
    const t0 = Math.floor(Math.random() * self.leapfrogSteps);

    // use leapfrog integration to find proposal
    let q = q0.copy();
    let p = p0.copy();
    const trajectory = [q.copy()];
    const neg_energies = [self.logDensity(q) - p.norm2() / 2]
    // integrate backwards
    for (let i = 0; i < t0; i++) {
      p.increment(self.gradLogDensity(q).scale(-self.dt / 2));
      q.increment(p.scale(-self.dt));
      p.increment(self.gradLogDensity(q).scale(-self.dt / 2));
      // add to beginning of trajectory
      trajectory.unshift(q.copy());
      // add negative energy
      neg_energies.unshift(self.logDensity(q) - p.norm2() / 2);
    }

    // reset to initial points
    q = q0.copy();
    p = p0.copy();
    
    // integrate forwards
    for (let i = 0; i < self.leapfrogSteps - t0; i++) {
      p.increment(self.gradLogDensity(q).scale(self.dt / 2));
      q.increment(p.scale(self.dt));
      p.increment(self.gradLogDensity(q).scale(self.dt / 2));
      // add to end of trajectory
      trajectory.push(q.copy());
      // add negative energy
      neg_energies.push(self.logDensity(q) - p.norm2() / 2);
    }

    // sample the point from trajectory
    let idx = Categorical.getSample(softmax(neg_energies))
    const q_chosen = trajectory[idx]

    // add integrated trajectory to visualizer animation queue
    visualizer.queue.push({
      type: "proposal",
      proposal: q_chosen,
      trajectory: trajectory,
      initial_index: t0,
      initialMomentum: p0,
    });

    // always accept
    self.chain.push(q_chosen.copy())
    visualizer.queue.push({ type: "accept", proposal: q_chosen })
  },
});
