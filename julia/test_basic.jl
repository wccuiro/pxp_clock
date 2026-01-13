using ITensors, ITensorMPS, TensorMixedStates, .Qubits
using DataFrames

Pdn = 0.5*(Id - Z)
L3m = (Pdn ⊗ Sm ⊗ Pdn)
L3p = (Pdn ⊗ Sp ⊗ Pdn)

hamiltonian(n,om) = sum(om*Pdn(i-1)*X(i)*Pdn(i+1) for i in 2:n-1) + om*Pdn(n)*X(1)*Pdn(2) + om*Pdn(n-1)*X(n)*Pdn(1)
dissipators(n,gm,gp) = sum(Dissipator(sqrt(gm)* L3m)(i-1, i, i+1) + Dissipator(sqrt(gp)*L3p)(i-1, i, i+1) for i in 2:n-1) + Dissipator(sqrt(gm)* L3m)(n, 1, 2) + Dissipator(sqrt(gp)*L3p)(n, 1, 2) + Dissipator(sqrt(gm)* L3m)(n-1, n, 1) + Dissipator(sqrt(gp)*L3p)(n-1, n, 1)

sim_data(n, om, gm, gp, step) = SimData(
    name = "Test2",
    phases = [
        CreateState(
            type = Mixed(),
            system = System(n, Qubit()),
            state = [ "Dn" for i in 1:n ]),
        Evolve(
            duration = 6,
            time_step = step,
            algo = Tdvp(),
            evolver = -im*hamiltonian(n,om) + dissipators(n, gm, gp),
            limits = Limits(cutoff = 1e-30, maxdim = 100),
            measures = [
                "occupation.dat" => sum(Proj("Up")(i) for i in 1:n),
                "correlation2.dat" => sum(Proj("Up")(i)*Proj("Up")(i+1) for i in 1:n-1) + Proj("Up")(n)*Proj("Up")(1),
                "correlation3.dat" => sum(Proj("Up")(i-1)*Proj("Dn")(i)*Proj("Up")(i+1) for i in 2:n-1) + Proj("Up")(n-1)*Proj("Dn")(n)*Proj("Up")(1) + Proj("Up")(n)*Proj("Dn")(1)*Proj("Up")(2)
            ]
        )
        # SteadyState(
        #     lindbladian = -im * hamiltonian(n,om) + dissipators(n,gm,gp),
        #     limits = Limits(cutoff = 1e-20, maxdim = [10, 20, 50]),
        #     nsweeps = 10,
        #     tolerance = 1e-5,
        #     measures = [
        #         "occupation.dat" => sum(Proj("Up")(i) for i in 1:n),
        #     ]
        # )
    ]
)

L = 20
omega = 0.0
gamma_m = 1.5
gamma_p = 1.0
time_step = 0.1


sim = runTMS(sim_data(L, omega, gamma_m, gamma_p, time_step))

println(sim)
# df = DataToFrame(sim)