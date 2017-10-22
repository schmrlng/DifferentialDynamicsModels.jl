__precompile__()

module DifferentialDynamicsModels

using StaticArrays
export AbstractState, State, AbstractControl, Control, DifferentialDynamics, CostFunctional
export Time, TimePlusQuadraticControl
export SteeringBVP, SteeringParams, SteeringCache, EmptySteeringParams, EmptySteeringCache
import Base.issymmetric
export Path, ControlInterval, ControlSequence
export StepControl, BVPControl, TimestampedState, InterpolationInterval
export SingleIntegratorDynamics
export state_dim, control_dim, duration, propagate, instantaneous_control, waypoints

abstract type AbstractState end
const State = Union{AbstractState, AbstractVector}
abstract type AbstractControl end
const Control = Union{AbstractControl, AbstractVector}
abstract type DifferentialDynamics end
abstract type CostFunctional end
struct Time <: CostFunctional end
struct TimePlusQuadraticControl{Du,T,DuDu} <: CostFunctional
    R::SMatrix{Du,Du,T,DuDu}
end

abstract type SteeringParams end
abstract type SteeringCache end
struct EmptySteeringParams <: SteeringParams end
struct EmptySteeringCache <: SteeringCache end
struct SteeringBVP{D<:DifferentialDynamics,C<:CostFunctional,SP<:SteeringParams,SC<:SteeringCache}
    dynamics::D
    cost::C
    params::SP
    cache::SC
end
function SteeringBVP(dynamics::DifferentialDynamics, cost::CostFunctional)
    SteeringBVP(dynamics, cost, EmptySteeringParams(), EmptySteeringCache())
end
function SteeringBVP(dynamics::DifferentialDynamics, cost::CostFunctional, params::SteeringParams)
    SteeringBVP(dynamics, cost, params, EmptySteeringCache())
end
issymmetric(bvp::SteeringBVP) = false
(bvp::SteeringBVP)(x0::State, xf::State, cost_bound::AbstractFloat) = bvp(x0, xf)    # general fallback

const Path{S} = AbstractVector{S} where {S<:State}
abstract type ControlInterval end
const ControlSequence{CI} = AbstractVector{CI} where {CI<:ControlInterval}
duration(cs::ControlSequence) = sum(duration(c) for c in cs)

propagate(f::DifferentialDynamics, x::State, cs::ControlSequence) = foldl((x,c) -> propagate(f,x,c), x, cs)
function propagate(f::DifferentialDynamics, x::State, cs::ControlSequence, s::AbstractFloat)
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
function propagate(f::DifferentialDynamics, x::State, c::ControlInterval, ss::AbstractVector)
    [propagate(f, x, c, s) for s in ss]
end
function propagate(f::DifferentialDynamics, x::State, cs::ControlSequence, ss::AbstractVector)
    issorted(ss) || error("Times should be sorted as input to propagate.")
    tf = duration(cs)
    ss = clamp.(ss, zero(tf), tf)
    path = typeof(x)[]
    t, i = zero(tf), 1
    for s in ss
        @inbounds c = cs[i]
        while s >= t + duration(c) && i < length(cs)    # second clause necessary because of numerical error
            x = propagate(f, x, c)
            t += duration(c)
            i += 1
            @inbounds c = cs[i]
        end
        push!(path, propagate(f, x, c, s-t))
    end
    path
end

function instantaneous_control(f::DifferentialDynamics, x::State, cs::ControlSequence, s::AbstractFloat)
    s <= 0 && return instantaneous_control(f, x, cs[1], zero(s))
    x_prev = x
    t = zero(s)
    for c in cs
        if s >= t + duration(c)
            x_prev = x
            x = propagate(f, x, c)
            t += duration(c)
        else
            return instantaneous_control(f, x, c, s-t)
        end
    end
    instantaneous_control(f, x_prev, cs[end], duration(cs[end]))
end
function instantaneous_control(f::DifferentialDynamics, x::State, c::ControlInterval, ss::AbstractVector)
    [instantaneous_control(f, x, c, s) for s in ss]
end
function instantaneous_control(f::DifferentialDynamics, x::State, cs::ControlSequence, ss::AbstractVector)
    issorted(ss) || error("Times should be sorted as input to instantaneous_control.")
    tf = duration(cs)
    ss = clamp.(ss, zero(tf), tf)
    ics = typeof(instantaneous_control(f, x, cs[1], zero(tf)))[]
    t, i = zero(tf), 1
    for s in ss
        @inbounds c = cs[i]
        while s >= t + duration(c) && i < length(cs)    # second clause necessary because of numerical error
            x = propagate(f, x, c)
            t += duration(c)
            i += 1
            @inbounds c = cs[i]
        end
        push!(ics, instantaneous_control(f, x, c, s-t))
    end
    ics
end

function waypoints(f::DifferentialDynamics, x::State, c::Union{ControlInterval, ControlSequence}, dt::AbstractFloat)
    propagate(f, x, c, 0:dt:oftype(dt, duration(c)))
end
function waypoints(f::DifferentialDynamics, x::State, c::Union{ControlInterval, ControlSequence}, N::Int)
    propagate(f, x, c, linspace(0, duration(c), N))
end

### Control Intervals
## Step Control
struct StepControl{N,T,S<:StaticVector{N,T}} <: ControlInterval
    t::T
    u::S
end
(::Type{SC})(t::T, u::S) where {N,T,S<:StaticVector{N,T},SC<:StepControl} = StepControl{N,T,S}(t, u)
duration(c::StepControl) = c.t
Base.zero(x::StepControl{N,T,S}) where {N,T,S} = StepControl(T(0), zeros(S))
Base.zero(x::Type{StepControl{N,T,S}}) where {N,T,S} = StepControl(T(0), zeros(S))

function propagate(f::DifferentialDynamics, x::State, c::StepControl, s::AbstractFloat)
    s <= 0 ? x : s >= duration(c) ? propagate(f, x, c) : propagate(f, x, StepControl(s,c.u))
end
function propagate_heun{T}(fn::Function, y0::AbstractArray{T}, Tf::T, Ti::T = T(0), N::Int = 10)
    dt = (Tf - Ti)/N
    y = y0
    t = Ti
    for i in 1:N
        k1 = dt*fn(y, t)
        k2 = dt*fn(y + k1, t + dt)
        y = y + (k1 + k2)/2
        t = t + dt
    end
    y
end
function propagate_rk4{T}(fn::Function, y0::AbstractArray{T}, Tf::T, Ti::T = T(0), N::Int = 10)
    dt = (Tf - Ti)/N
    y = y0
    t = Ti
    for i in 1:N
        k1 = dt*fn(y, t)
        k2 = dt*fn(y + k1/2, t + dt/2)
        k3 = dt*fn(y + k2/2, t + dt/2)
        k4 = dt*fn(y + k3, t + dt)
        y = y + (k1 + 2*k2 + 2*k3 + k4)/6
        t = t + dt
    end
    y
end
function propagate_ODE(f::DifferentialDynamics, x::State, c::StepControl, N::Int = 10)
    typeof(x)(c.t > 0 ? propagate_rk4((y,t) -> f(y, c.u), x, c.t, zero(c.t), N) :
                        propagate_rk4((y,t) -> -f(y, c.u), x, -c.t, zero(c.t), N))
end
propagate(f::DifferentialDynamics, x::State, c::StepControl) = propagate_ODE(f, x, c)    # general fallback
instantaneous_control(f::DifferentialDynamics, x::State, c::StepControl, s::AbstractFloat) = c.u

## BVP Control
struct BVPControl{T,S<:State,Fx<:Function,Fu<:Function} <: ControlInterval
    t::T
    y::S
    x::Fx
    u::Fu
end
duration(c::BVPControl) = c.t

propagate(f::DifferentialDynamics, x::State, c::BVPControl) = c.x
propagate(f::DifferentialDynamics, x::State, c::BVPControl, s::AbstractFloat) = c.x(x, c.y, c.t, s)
instantaneous_control(f::DifferentialDynamics, x::State, c::BVPControl, s::AbstractFloat) = c.u(x, c.y, c.t, s)

## State Interpolation Interval
struct TimestampedState{T,S<:State}
    t::T
    x::S
end
struct InterpolationInterval{A<:AbstractVector{TS} where {TS<:TimestampedState}}
    txs::A
end
duration(c::InterpolationInterval) = c.txs[end].t

propagate(f::DifferentialDynamics, x::State, c::InterpolationInterval) = c.txs[end].x
function propagate(f::DifferentialDynamics, x::State, c::InterpolationInterval, s::AbstractFloat)
    s <= 0 && return x
    i = findfirst(tx -> (tx.t >= s), c.txs)
    if i == 0
        return c.txs[end].x
    elseif i == 1
        @inbounds α = s/c.txs[1].t
        @inbounds return (1 - α)*x + α*c.txs[1].x
    else
        @inbounds α = (s - c.txs[i-1].t)/(c.txs[i].t - c.txs[i-1].t)
        @inbounds return (1 - α)*c.txs[i-1].x + α*c.txs[i].x
    end
end
function instantaneous_control(f::DifferentialDynamics, x::State, c::InterpolationInterval, s::AbstractFloat)
    error("instantaneous_control makes no sense for InterpolationInterval.")
    # Maybe InterpolationInterval should use TimestampedStateAndControl? This case should be covered by BVPControl.
end

### Single Integrator
struct SingleIntegratorDynamics{N} <: DifferentialDynamics end

state_dim(::SingleIntegratorDynamics{N}) where {N} = N
control_dim(::SingleIntegratorDynamics{N}) where {N} = N

function propagate(f::SingleIntegratorDynamics{N}, x::StaticVector{N,T}, c::StepControl{N,T}) where {N,T}
    x + c.t*typeof(x)(c.u)
end
(::SingleIntegratorDynamics{N})(x::StaticVector{N}, u::StaticVector{N}) where {N} = u

issymmetric(bvp::SteeringBVP{D,C,EmptySteeringParams}) where {D<:SingleIntegratorDynamics,C} = true
function (bvp::SteeringBVP{SingleIntegratorDynamics{N},Time})(x0::StaticVector{N,T},
                                                              xf::StaticVector{N,T}) where {N,T}
    c = norm(xf - x0)
    ctrl = StepControl(c, inv(c)*(xf - x0))
    return c, ctrl
end

end # module