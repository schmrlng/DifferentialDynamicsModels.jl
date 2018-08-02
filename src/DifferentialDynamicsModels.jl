module DifferentialDynamicsModels

using LinearAlgebra
using StaticArrays

export @maintain_type
export AbstractState, State, AbstractControl, Control, DifferentialDynamics, CostFunctional
export Time, TimePlusQuadraticControl
export StateSequence, ControlInterval, ControlSequence
export StepControl, RampControl, BVPControl
export SteeringBVP, SteeringConstraints, SteeringCache, EmptySteeringConstraints, EmptySteeringCache
export SingleIntegratorDynamics, BoundedControlNorm
export state_dim, control_dim, duration, propagate, instantaneous_control, waypoints
import Base: zero, getindex
import LinearAlgebra: issymmetric
export issymmetric

include("utils.jl")

# States, Controls, Dynamics, and Cost Functionals
abstract type AbstractState end
const State = Union{AbstractState, AbstractVector}
abstract type AbstractControl end
const Control = Union{AbstractControl, AbstractVector}
abstract type DifferentialDynamics end
abstract type CostFunctional end
struct Time <: CostFunctional end
struct TimePlusQuadraticControl{Du,TR<:SMatrix{Du,Du}} <: CostFunctional
    R::TR
end

# State/Control Sequences
const StateSequence{S} = AbstractVector{S} where {S<:State}
abstract type ControlInterval end
const ControlSequence{CI} = AbstractVector{CI} where {CI<:ControlInterval}
duration(cs::ControlSequence) = sum(duration(c) for c in cs)
zero(::CI) where {CI<:ControlInterval} = zero(CI)

## Propagation (state as a function of time)
propagate(f::DifferentialDynamics, x::State, cs::ControlSequence) = foldl((x,c) -> propagate(f,x,c), cs, init=x)
propagate(f::DifferentialDynamics, x::State, c::ControlInterval, ss) = map(s -> propagate(f, x, c, s), ss)
function propagate(f::DifferentialDynamics, x::State, cs::ControlSequence, s::Number)
    s <= 0 && return x
    t = zero(s)
    for c in cs
        if s >= t + duration(c)
            x = propagate(f, x, c)
            t += duration(c)
        else
            return propagate(f, x, c, s-t)
        end
    end
    x
end
function propagate(f::DifferentialDynamics, x::State, cs::ControlSequence, ss)
    tf = duration(cs)
    path = typeof(x)[]
    prev_s, t, i = zero(tf), zero(tf), 1
    for s in ss
        s = clamp(s, zero(tf), tf)
        @assert s >= prev_s "Input times must be sorted."
        @inbounds c = cs[i]
        while s >= t + duration(c) && i < length(cs)    # second clause necessary because of numerical error
            x = propagate(f, x, c)
            t += duration(c)
            i += 1
            @inbounds c = cs[i]
        end
        push!(path, propagate(f, x, c, s-t))
        prev_s = s
    end
    path
end

## Instantaneous Controls (control as a function of time)
instantaneous_control(c::ControlInterval, ss) = map(s -> instantaneous_control(c, s), ss)
function instantaneous_control(cs::ControlSequence, s::Number)
    s <= 0 && return instantaneous_control(cs[1], zero(s))
    t = zero(s)
    for c in cs
        if s >= t + duration(c)
            t += duration(c)
        else
            return instantaneous_control(c, s-t)
        end
    end
    instantaneous_control(c[end], duration(c[end]))
end
function instantaneous_control(cs::ControlSequence, ss)
    tf = duration(cs)
    ics = typeof(instantaneous_control(cs[1], zero(tf)))[]
    prev_s, t, i = zero(tf), zero(tf), 1
    for s in ss
        s = clamp(s, zero(tf), tf)
        @assert s >= prev_s "Input times must be sorted."
        @inbounds c = cs[i]
        while s >= t + duration(c) && i < length(cs)    # second clause necessary because of numerical error
            t += duration(c)
            i += 1
            @inbounds c = cs[i]
        end
        push!(ics, instantaneous_control(c, s-t))
        prev_s = s
    end
    ics
end

## Waypoints (convenience methods for state propagation)
function waypoints(f::DifferentialDynamics, x::State, c::Union{ControlInterval, ControlSequence}, dt::AbstractFloat)
    propagate(f, x, c, 0:dt:oftype(dt, duration(c)))
end
function waypoints(f::DifferentialDynamics, x::State, c::Union{ControlInterval, ControlSequence}, N::Int)
    propagate(f, x, c, range(0, stop=duration(c), length=N))
end

# Control Intervals
function propagate_ode(f::DifferentialDynamics, x::State, c::ControlInterval, s::Number=duration(c); N=10)
    s > 0 ? ode_rk4((y,t) -> f(y, instantaneous_control(c, t)), x, s, zero(s), N) :
            ode_rk4((y,t) -> -f(y, instantaneous_control(c, -t)), x, -s, zero(s), N)
end
propagate(f::DifferentialDynamics, x::State, c::ControlInterval) = propagate_ode(f, x, c)    # general fallback

## Step Control
struct StepControl{N,T,S<:StaticVector{N}} <: ControlInterval
    t::T
    u::S
    function (::Type{SC})(t::T, u::S) where {N,T,S<:StaticVector{N},SC<:StepControl}
        new{N,T,S}(t, u)
    end
end
const ZeroOrderHoldControl{N,T,S} = StepControl{N,T,S}
duration(c::StepControl) = c.t
zero(x::Type{StepControl{N,T,S}}) where {N,T,S} = StepControl(T(0), zeros(S))
getindex(c::StepControl, i) = StepControl(c.t, c.u[i])
function propagate(f::DifferentialDynamics, x::State, c::StepControl, s::Number)
    s <= 0           ? x :
    s >= duration(c) ? propagate(f, x, c) :
                       propagate(f, x, StepControl(s, c.u))
end
instantaneous_control(c::StepControl, s::Number) = c.u

## Ramp Control
struct RampControl{N,T,S0<:StaticVector{N},Sf<:StaticVector{N}} <: ControlInterval
    t::T
    u0::S0
    uf::Sf
    function (::Type{RC})(t::T, u0::S0, uf::Sf) where {N,T,S0<:StaticVector{N},Sf<:StaticVector{N},RC<:RampControl}
        new{N,T,S0,Sf}(t, u0, uf)
    end
end
const FirstOrderHoldControl{N,T,S0,Sf} = RampControl{N,T,S0,Sf}
RampControl(c::StepControl) = RampControl(c.t, c.u, c.u)
duration(c::RampControl) = c.t
zero(x::Type{RampControl{N,T,S0,Sf}}) where {N,T,S0,Sf} = RampControl(T(0), zeros(S0), zeros(Sf))
getindex(c::RampControl, i) = RampControl(c.t, c.u0[i], c.uf[i])
function propagate(f::DifferentialDynamics, x::State, c::RampControl, s::Number)
    s <= 0           ? x :
    s >= duration(c) ? propagate(f, x, c) :
                       propagate(f, x, RampControl(s, c.u0, instantaneous_control(c, s)))
end
instantaneous_control(c::RampControl, s::Number) = c.u0 + (s/c.t)*(c.uf - c.u0)

## BVP Control
struct BVPControl{T,S0<:State,Sf<:State,Fx<:Function,Fu<:Function} <: ControlInterval
    t::T
    x0::S0
    xf::Sf
    x::Fx
    u::Fu
end
duration(c::BVPControl) = c.t
propagate(f::DifferentialDynamics, x::State, c::BVPControl) = (x - c.x0) + c.xf
propagate(f::DifferentialDynamics, x::State, c::BVPControl, s::Number) = (x - c.x0) + c.x(c.x0, c.xf, c.t, s)
instantaneous_control(f::DifferentialDynamics, x::State, c::BVPControl, s::Number) = c.u(c.x0, c.xf, c.t, s)

# Steering Two-Point Boundary Value Problems (BVPs)
abstract type SteeringConstraints end
abstract type SteeringCache end
struct EmptySteeringConstraints <: SteeringConstraints end
struct BoundedControlNorm{P,T} <: SteeringConstraints
    b::T
end
BoundedControlNorm(b::T=1) where {T} = BoundedControlNorm{2,T}(b)
BoundedControlNorm{P}(b::T) where {P,T} = BoundedControlNorm{P,T}(b)
struct EmptySteeringCache <: SteeringCache end
struct SteeringBVP{D<:DifferentialDynamics,C<:CostFunctional,SC<:SteeringConstraints,SD<:SteeringCache}
    dynamics::D
    cost::C
    constraints::SC
    cache::SD
end
function SteeringBVP(dynamics::DifferentialDynamics, cost::CostFunctional;
                     constraints::SteeringConstraints=EmptySteeringConstraints(),
                     cache::SteeringCache=EmptySteeringCache())
    SteeringBVP(dynamics, cost, constraints, cache)
end
issymmetric(bvp::SteeringBVP) = false                                         # general fallback
(bvp::SteeringBVP)(x0::State, xf::State, cost_bound::Number) = bvp(x0, xf)    # general fallback

# Single Integrator
struct SingleIntegratorDynamics{N} <: DifferentialDynamics end

state_dim(::SingleIntegratorDynamics{N}) where {N} = N
control_dim(::SingleIntegratorDynamics{N}) where {N} = N

(::SingleIntegratorDynamics{N})(x::StaticVector{N}, u::StaticVector{N}) where {N} = u
propagate(f::SingleIntegratorDynamics{N}, x::StaticVector{N}, c::StepControl{N}) where {N} = x + c.t*c.u
propagate(f::SingleIntegratorDynamics{N}, x::StaticVector{N}, c::RampControl{N}) where {N} = x + c.t*(c.u0 + c.uf)/2

issymmetric(bvp::SteeringBVP{<:SingleIntegratorDynamics,<:CostFunctional,<:BoundedControlNorm}) = true
function (bvp::SteeringBVP{SingleIntegratorDynamics{N},Time,<:BoundedControlNorm{2}})(x0::StaticVector{N},
                                                                                      xf::StaticVector{N}) where {N}
    c = norm(xf - x0)/bvp.constraints.b
    ctrl = StepControl(c, SVector((xf - x0)/c))    # convert to SVector otherwise control will maintain State type
    (cost=c, controls=ctrl)
end

end # module
