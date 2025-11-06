using ITensors, ITensorMPS, TensorMixedStates, .Qubits

hamiltonian(n,om) = sum(om*Sm(i-1)*Sp(i-1)*X(i)*Sm(i+1)*Sp(i+1) for i in 2:n-1) + om*Sm(n)*Sp(n)*X(1)*Sm(2)*Sp(2) + om*Sm(n-1)*Sp(n-1)*X(n)*Sm(1)*Sp(1)
dissipators(n,gm,gp) = sum(Dissipator(sqrt(gm)*Sm*Sp)(i-1)*Dissipator(Sm)(i)*Dissipator(sqrt(gm)*Sm*Sp)(i+1) + Dissipator(sqrt(gp)*Sm*Sp)(i-1)*Dissipator(Sp)(i)*Dissipator(Sm*Sp)(i+1) for i in 2:n-1) + Dissipator(sqrt(gp)*Sm*Sp)(n)*Dissipator(Sp)(1)*Dissipator(Sm*Sp)(2) + Dissipator(sqrt(gp)*Sm*Sp)(n-1)*Dissipator(Sp)(n)*Dissipator(Sm*Sp)(1) + Dissipator(sqrt(gm)*Sm*Sp)(n)*Dissipator(Sm)(1)*Dissipator(Sm*Sp)(2) + Dissipator(sqrt(gm)*Sm*Sp)(n-1)*Dissipator(Sm)(n)*Dissipator(Sm*Sp)(1)

sim_data(n, om, gm, gp, step) = SimData(
    name = "Test",
    phases = [
        CreateState(
            type = Mixed(),
            system = System(n, Qubit()),
            state = [ "Dn" for i in 1:n ]),
        Evolve(
            duration = 4,
            time_step = step,
            algo = Tdvp(),
            evolver = -im*hamiltonian(n,om) + dissipators(n, gm, gp),
            limits = Limits(cutoff = 1e-30, maxdim = 100),
            measures = [
                "density.dat" => Sz
            ]
        )
    ]
)

runTMS(sim_data(10, 1., 1., 1.5, 0.05))