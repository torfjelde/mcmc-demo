"use strict";

MCMC.registerAlgorithm("CoupledMetropolisHastingsHMC", {
  description: "Coupled Metropolis-Hastings Hamiltonian Monte Carlo",

  about: () => {
    window.open("https://en.wikipedia.org/wiki/Hybrid_Monte_Carlo");
  },

  init: (self) => {
    self.leapfrogSteps = 37;
    self.dt = 0.1;
  },

  reset: (self) => {
    self.chain_1 = [MultivariateNormal.getSample(self.dim)];
    self.chain_2 = [MultivariateNormal.getSample(self.dim)];
    self.chains = [self.chain_1, self.chain_2];
    self.chain = self.chain_1;
  },

  attachUI: (self, folder) => {
    folder.add(self, "leapfrogSteps", 5, 100).step(1).name("Leapfrog Steps");
    folder.add(self, "dt", 0.05, 0.5).step(0.025).name("Leapfrog &Delta;t");
    folder.open();
  },

  step: (self, visualizer) => {
    const q0_1 = self.chain.last();
    const q0_2 = self.chain_2.last();
    const p0 = MultivariateNormal.getSample(self.dim);

    // use leapfrog integration to find proposal
    // 1st trajectory
    const q_1 = q0_1.copy();
    const trajectory_1 = [q_1.copy()];
    const p_1 = p0.copy();
    for (let i = 0; i < self.leapfrogSteps; i++) {
      p_1.increment(self.gradLogDensity(q_1).scale(self.dt / 2));
      q_1.increment(p_1.scale(self.dt));
      p_1.increment(self.gradLogDensity(q_1).scale(self.dt / 2));
      trajectory_1.push(q_1.copy());
    }

    // 2nd trajectory
    const q_2 = q0_2.copy();
    const trajectory_2 = [q_2.copy()];
    const p_2 = p0.copy();
    for (let i = 0; i < self.leapfrogSteps; i++) {
      p_2.increment(self.gradLogDensity(q_2).scale(self.dt / 2));
      q_2.increment(p_2.scale(self.dt));
      p_2.increment(self.gradLogDensity(q_2).scale(self.dt / 2));
      trajectory_2.push(q_2.copy());
    }

    // // add integrated trajectory to visualizer animation queue
    // visualizer.queue.push({
    //   type: "proposal",
    //   proposal: q_1,
    //   trajectory: trajectory_1,
    //   initialMomentum: p0,
    // });

    // add integrated trajectory to visualizer animation queue
    visualizer.queue.push({
      type: "proposal",
      proposal: q_1,  // TODO: Make `proposals` maybe.
      trajectories: [trajectory_1, trajectory_2],
      initialMomenta: [p0, p0],
    });


    // calculate acceptance ratio
    const H0_1 = -self.logDensity(q0_1) + p0.norm2() / 2;
    const H_1 = -self.logDensity(q_1) + p_1.norm2() / 2;
    const logAcceptRatio_1 = -H_1 + H0_1;

    const H0_2 = -self.logDensity(q0_2) + p0.norm2() / 2;
    const H_2 = -self.logDensity(q_2) + p_2.norm2() / 2;
    const logAcceptRatio_2 = -H_2 + H0_2;


    // accept or reject proposal
    const u = Math.random()

    if (u < Math.exp(logAcceptRatio_1)) {
      self.chain_1.push(q_1.copy());
      visualizer.queue.push({ type: "accept", proposal: q_1, from: q0_1 });
    } else {
      self.chain_1.push(q0_1.copy());
      visualizer.queue.push({ type: "reject", proposal: q_1, from: q0_1 });
    }

    if (u < Math.exp(logAcceptRatio_2)) {
      self.chain_2.push(q_2.copy());
      visualizer.queue.push({ type: "accept", proposal: q_2, from: q0_2 });
    } else {
      self.chain_2.push(q0_2.copy());
      visualizer.queue.push({ type: "reject", proposal: q_2, from: q0_2 });
    }
  },
});
