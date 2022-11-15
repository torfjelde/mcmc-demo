"use strict";

MCMC.registerAlgorithm("CoupledMultinomialHMC", {
  description: "Coupled Multinomial Hamiltonian Monte Carlo",

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
    const q0_1 = self.chain_1.last();
    const q0_2 = self.chain_2.last();
    const p0 = MultivariateNormal.getSample(self.dim);

    // sample index of initial point uniformly
    const t0 = Math.floor(Math.random() * self.leapfrogSteps);

    // use leapfrog integration to find proposal
    let q_1 = q0_1.copy();
    let p_1 = p0.copy();
    const trajectory_1 = [q_1.copy()];
    const neg_energies_1 = [self.logDensity(q_1) - p_1.norm2() / 2]
    // integrate backwards
    for (let i = 0; i < t0; i++) {
      p_1.increment(self.gradLogDensity(q_1).scale(-self.dt / 2));
      q_1.increment(p_1.scale(-self.dt));
      p_1.increment(self.gradLogDensity(q_1).scale(-self.dt / 2));
      // add to beginning of trajectory
      trajectory_1.unshift(q_1.copy());
      // add negative energy
      neg_energies_1.unshift(self.logDensity(q_1) - p_1.norm2() / 2);
    }

    // reset to initial points
    q_1 = q0_1.copy();
    p_1 = p0.copy();
    
    // integrate forwards
    for (let i = 0; i < self.leapfrogSteps - t0; i++) {
      p_1.increment(self.gradLogDensity(q_1).scale(self.dt / 2));
      q_1.increment(p_1.scale(self.dt));
      p_1.increment(self.gradLogDensity(q_1).scale(self.dt / 2));
      // add to end of trajectory
      trajectory_1.push(q_1.copy());
      // add negative energy
      neg_energies_1.push(self.logDensity(q_1) - p_1.norm2() / 2);
    }

    // use leapfrog integration to find proposal
    let q_2 = q0_2.copy();
    let p_2 = p0.copy();
    const trajectory_2 = [q_2.copy()];
    const neg_energies_2 = [self.logDensity(q_2) - p_2.norm2() / 2]
    // integrate backwards
    for (let i = 0; i < t0; i++) {
      p_2.increment(self.gradLogDensity(q_2).scale(-self.dt / 2));
      q_2.increment(p_2.scale(-self.dt));
      p_2.increment(self.gradLogDensity(q_2).scale(-self.dt / 2));
      // add to beginning of trajectory
      trajectory_2.unshift(q_2.copy());
      // add negative energy
      neg_energies_2.unshift(self.logDensity(q_2) - p_2.norm2() / 2);
    }

    // reset to initial points
    q_2 = q0_2.copy();
    p_2 = p0.copy();
    
    // integrate forwards
    for (let i = 0; i < self.leapfrogSteps - t0; i++) {
      p_2.increment(self.gradLogDensity(q_2).scale(self.dt / 2));
      q_2.increment(p_2.scale(self.dt));
      p_2.increment(self.gradLogDensity(q_2).scale(self.dt / 2));
      // add to end of trajectory
      trajectory_2.push(q_2.copy());
      // add negative energy
      neg_energies_2.push(self.logDensity(q_2) - p_2.norm2() / 2);
    }

    // sample the point from trajectory
    let [idx_1, idx_2] = MaximalCouplingCategorical.getSample(
      softmax(neg_energies_1),
      softmax(neg_energies_2)
    );

    const q_chosen_1 = trajectory_1[idx_1];
    const q_chosen_2 = trajectory_2[idx_2];


    // add integrated trajectory to visualizer animation queue
    visualizer.queue.push({
      type: "proposal",
      proposal: q_chosen_1,
      trajectories: [trajectory_1, trajectory_2],
      initial_indices: [t0, t0],
      initialMomenta: [p0, p0],
    });

    // always accept
    self.chain_1.push(q_chosen_1.copy())
    visualizer.queue.push({ type: "accept", proposal: q_chosen_1, from: q0_1 });

    // TODO: Support multiple chains properly.
    self.chain_2.push(q_chosen_2.copy())
    visualizer.queue.push({ type: "accept", proposal: q_chosen_2, from: q0_2 });
  },
});
