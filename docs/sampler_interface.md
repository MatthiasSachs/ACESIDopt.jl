# Unified Sampler Interface

The MSamplers module now provides a unified interface for MCMC sampling with configurable sampler structures.

## Sampler Structures

### RWMCSampler

Random Walk Monte Carlo sampler with symmetric proposals.

```julia
RWMCSampler(; n_samples=1000, burnin=1000, thin=10)
```

**Parameters:**
- `n_samples`: Number of samples to collect (after burnin and thinning)
- `burnin`: Number of initial steps to discard
- `thin`: Keep every thin-th sample

### MALASampler

Metropolis-Adjusted Langevin Algorithm sampler using gradient information.

```julia
MALASampler(; n_samples=1000, burnin=1000, thin=10, collect_forces=false)
```

**Parameters:**
- `n_samples`: Number of samples to collect (after burnin and thinning)
- `burnin`: Number of initial steps to discard
- `thin`: Keep every thin-th sample
- `collect_forces`: Whether to collect forces in the trajectory

## Unified Interface

Both samplers use the same `run_sampler` function:

```julia
run_sampler(sampler, initial_system, model, T, step_size)
```

**Parameters:**
- `sampler`: Either `RWMCSampler` or `MALASampler` instance
- `initial_system`: Starting atomic configuration
- `model`: Potential model (e.g., ACE potential, HarmonicCalculator)
- `T`: Temperature in Kelvin
- `step_size`: Step size in Ångströms

**Returns:**
- `samples`: Vector of sampled configurations
- `acceptance_rate`: Fraction of accepted moves
- `traj`: Trajectory data (energy, and optionally forces)

## Examples

### RWMC Sampling

```julia
using ACESIDopt: HarmonicCalculator, MSamplers
using ACESIDopt.MSamplers: RWMCSampler, run_sampler

# Create sampler configuration
rwmc = RWMCSampler(
    n_samples=5000,
    burnin=2000,
    thin=5
)

# Create system and model
system = # ... your atomic system
calc = HarmonicCalculator(:Si, 0.1)

# Run sampling
samples, acc_rate, traj = run_sampler(rwmc, system, calc, 300.0, 0.1)

println("Acceptance rate: $(acc_rate)")
println("Collected $(length(samples)) samples")
```

### MALA Sampling

```julia
using ACESIDopt: HarmonicCalculator, MSamplers
using ACESIDopt.MSamplers: MALASampler, run_sampler

# Create sampler configuration with force collection
mala = MALASampler(
    n_samples=5000,
    burnin=2000,
    thin=5,
    collect_forces=true
)

# Create system and model
system = # ... your atomic system
calc = HarmonicCalculator(:Si, 0.1)

# Run sampling
samples, acc_rate, traj = run_sampler(mala, system, calc, 300.0, 0.1)

println("Acceptance rate: $(acc_rate)")
println("Collected $(length(samples)) samples")
println("Force samples: $(length(traj.forces))")
```

## Backward Compatibility

The original functions are still available for backward compatibility:

```julia
# RWMC (old interface)
samples, acc_rate, traj = run_rwmc_sampling(
    system, model, n_samples, T;
    step_size=step_size,
    burnin=burnin,
    thin=thin
)

# MALA (old interface)
samples, acc_rate, traj = run_mala_sampling(
    system, model, n_samples, T;
    step_size=step_size,
    burnin=burnin,
    thin=thin,
    collect_forces=false
)
```

## Advantages of the New Interface

1. **Type safety**: Sampler configuration is validated at construction time
2. **Cleaner code**: Separate configuration from execution
3. **Extensibility**: Easy to add new sampler types
4. **Consistent API**: Same interface for all samplers
5. **Better documentation**: Sampler parameters are self-documenting
