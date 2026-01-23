module MFitmodel

using LinearAlgebra: Diagonal
using ACEpotentials: _make_prior, make_atoms_data, assess_dataset, 
                     _rep_dimer_data_atomsbase, default_weights, AtomsData
using ACEpotentials.Models: set_linear_parameters!, set_committee!
using ACEfit
using AtomsBase: AbstractSystem

"""
   acefit!(rawdata, model; kwargs...)

provides a convenient interface to fitting the parameters of an ACE model. 
The data should be provided as a collection of `AbstractSystem` structures. 

Keyword arguments:
* `energy_key`, `force_key`, `virial_key` specify 
the label of the data to which the parameters will be fitted. 
* `weights` specifies the regression weights, default is 30 for energy, 1 for forces and virials
* `solver` specifies the lsq solver, default is `BLR` (BayesianLinearRegression)
* `smoothness` specifies the smoothness prior, i.e. how strongly damped 
   parameters corresponding to high polynomial degrees are; is 2.
* `prior` specifies a covariance of the prior, if `nothing` then a smoothness prior 
   is used, using the `smoothness` parameter 
* `repulsion_restraint` specifies whether to add artificial data to the training 
   set that effectively introduces a restraints encouraging repulsion 
   in the limit rij -> 0.
* `restraint_weight` specifies the weight of the repulsion restraint.
* `export_lammps` : path to a file to which the fitted potential will be exported 
   in a LAMMPS compatible format (yace)
* `export_json` : path to a file to which the fitted potential will be exported 
   in a JSON format, which can be read from Julia or Python
"""
function acefit(raw_data::AbstractArray{<: AbstractSystem}, model;
                validation_set = nothing, 
                solver = ACEfit.BLR(),
                weights = default_weights(),
                energy_key = "energy", 
                force_key = "force", 
                virial_key = "virial", 
                smoothness = 4, 
                prior = nothing, 
                repulsion_restraint = false, 
                restraint_weight = 0.01, 
                export_lammps = nothing, 
                export_json = nothing, 
                verbose=true,
                kwargs...
   )

   data = make_atoms_data(raw_data, model; 
                          energy_key = energy_key, 
                          force_key = force_key, 
                          virial_key = virial_key, 
                          weights = weights)

   # print some information about the dataset 
   # (how many observations in each class)
   if verbose
      assess_dataset(
         data;
         energy_key = energy_key, 
         force_key  = force_key, 
         virial_key = virial_key,
         kwargs...
      )
   end 

   if repulsion_restraint 
      restraint_data = _rep_dimer_data_atomsbase(
               model; 
               weight = restraint_weight,
               energy_key = Symbol(energy_key),
               kwargs...
            )
      append!(data, restraint_data)
      # return nothing 
      # error("Repulsion restraint is currently not implemented")
      # if eltype(data) == AtomsData
      #    append!(data, _rep_dimer_data(model; weight = restraint_weight))
      # else
      #    tmp = _rep_dimer_data_atomsbase(
      #       model; 
      #       weight = restraint_weight,
      #       energy_key = Symbol(energy_key),
      #       kwargs...
      #       )
      #    append!(data, tmp)
      # end
   end
               
   # build a prior from the kw arguments 
   P = _make_prior(model, smoothness, prior)

   # actual assembly of the least square system 
   A, Y, W = ACEfit.assemble(data, model)

   # transform the system to incorporate the prior and the weights 
   # then solve the transformed problem 
   Ap = Diagonal(W) * (A / P) 
   Y = W .* Y

   if isnothing(validation_set) 
      result = ACEfit.solve(solver, Ap, Y)

   else
      @info("assemble validation system")
      val_data = make_atoms_data(validation_set, model; 
                                 energy_key = energy_key, 
                                 force_key = force_key, 
                                 virial_key = virial_key, 
                                 weights = weights)
      Av, Yv, Wv = ACEfit.assemble(val_data, model)
      Avp = Diagonal(Wv) * (Av / P)
      Yv = Wv .* Yv

      result = ACEfit.solve(solver, Ap, Y, Avp, Yv)
   end
   
   coeffs = P \ result["C"]   

   # dispatch setting of parameters 
   set_linear_parameters!(model, coeffs)

   if haskey(result, "committee")
      co_coeffs = result["committee"]
      co_ps_vec = [ P \ co_coeffs[:,i] for i in 1:size(co_coeffs,2) ]
      set_committee!(model, co_ps_vec)
   end

   if export_lammps != nothing 
      error("automatic lammps export currently not supported")
      # export2lammps(export_lammps, model)
   end
   if export_json != nothing 
      error("automatic json export currently not supported")
      # export2json(export_json, model)
   end

   return model, result 
end



function compute_errors(raw_data::AbstractArray{<: AbstractSystem}, model; 
                       energy_key = "energy", 
                       force_key = "force", 
                       virial_key = "virial", 
                       weights = default_weights(), 
                       verbose = true,
                       return_efv = false, 
                       )
   data = [ AtomsData(at; energy_key = energy_key, force_key=force_key, 
                          virial_key = virial_key, weights = weights, 
                          v_ref = nothing)
            for at in raw_data ]
   return compute_errors(data, model; verbose=verbose, return_efv = return_efv)
end

end