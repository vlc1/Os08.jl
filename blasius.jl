### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ 82c64cc3-81a7-45d8-87b4-81f28b8c0a9d
using LinearAlgebra, NLsolve, Plots

# ╔═╡ dc4e617c-78d2-4c38-b03d-971a7a82c790
md"""
# Prérequis

!!! warning "Remarque"

	Les cellules suivantes ont été copiées du TP précédent. Merci de ne pas les modifier.

	À la différence des deux derniers TP, celui-ci ne sera pas évalué mais réalisé en classe ensemble.

"""

# ╔═╡ cd52be0a-6128-4e25-ab0c-885620fefcbd
explicit!(res, x, y, τ, f, t) = res .= x - y - τ * f(t, y)

# ╔═╡ ebf51cb6-dce4-457d-b94e-cd3c382905fe
implicit!(res, x, y, τ, f, t) = res .= x - y - τ * f(t + τ, x)

# ╔═╡ 8c7a1da1-d048-4724-9375-31adc372dbb1
midpoint!(res, x, y, τ, f, t) = res .= x - y - τ * f(t + τ / 2, (x + y) / 2)

# ╔═╡ 0d54c141-0114-457d-8332-1551cb8984ec
rk1! = explicit!

# ╔═╡ ea19915a-76d5-4ad1-97fd-e096aa38aa90
function rk2!(res, x, y, τ, f, t)
	k₁ = f(t, y)
	k₂ = f(t + τ / 2, y + τ * k₁ / 2)
	res .= x - y - τ * k₂
end

# ╔═╡ 1624bbe7-ebd5-415e-9a76-dbbea721124d
function rk3!(res, x, y, τ, f, t)
	k₁ = f(t, y)
	k₂ = f(t + τ / 3, y + τ * k₁ / 3)
	k₃ = f(t + 2τ / 3, y + 2τ * k₂ / 3)
	res .= x - y - τ * (k₁ + 3k₃) / 4
end

# ╔═╡ acd1c234-f035-4f04-bf1a-2ec357657117
function rk4!(res, x, y, τ, f, t)
	k₁ = f(t, y)
	k₂ = f(t + τ / 2, y + τ * k₁ / 2)
	k₃ = f(t + τ / 2, y + τ * k₂ / 2)
	k₄ = f(t + τ, y + τ * k₃)
	res .= x - y - τ * (k₁ + 2k₂ + 2k₃ + k₄) / 6
end

# ╔═╡ 983d360a-ac0c-4e4b-a4d6-55f3d80f43b5
md"""
# L'équation de Blasius

Résoudre l'équation de Blasius
```math
2f''' \left ( x \right ) + f'' \left ( x \right ) f \left ( x \right ) = 0, \quad x > 0,
```
avec les conditions aux limites
```math
\left \{ \begin{aligned}
f \left ( 0 \right ) & = 0, \\
f' \left ( 0 \right ) & = 0, \\
f' \left ( \infty \right ) & = 1.
\end{aligned} \right .
```

"""

# ╔═╡ 30bf3035-ca50-4086-ab2e-888aed6eadf7
function cauchy(scheme!, f, τ, s, y₀, t₀ = zero(τ))
	t, y = t₀, y₀
    T, Y = [t], [y]

	while t < (1 - √eps(t)) * s
		y = getproperty(
			nlsolve(y) do res, x
				scheme!(res, x, y, τ, f, t)
			end,
			:zero)
        t += τ

        push!(Y, y)
        push!(T, t)
	end

	T, Y
end

# ╔═╡ b6cd4932-c551-406e-8ba6-ee93f9c436c8
function source(t, y)
	x = similar(y)
	x[1] = y[2]
	x[2] = y[3]
	x[3] = -y[1] * y[3] / 2
	return x
end

# ╔═╡ 44b3bef6-f820-492f-8d01-6792d7495759
md"""
Mathématiquement, la manière la plus générale d'écrire une condition aux limites est sous la forme
```math
h \left ( u \left ( a \right ), u \left ( b \right ) \right ) = 0
```
où
```math
h \colon \left \vert \begin{aligned}
\mathbb{R} ^ n \times \mathbb{R} ^n & \to \mathbb{R} ^ n, \\
\left ( u, v \right ) & \mapsto h \left ( u, v \right ).
\end{aligned} \right .
```

## Exemple : la conduction en régime stationnaire avec des conditions de Dirichlet

```math
\left \{ \begin{aligned}
T \left ( a \right ) & = \pi, \\
T' \left ( b \right ) & = -1
\end{aligned} \right .
```
alors
```math
h \colon \left \vert \begin{aligned}
\mathbb{R} ^ 2 \times \mathbb{R} ^ 2 & \to \mathbb{R} ^ 2, \\
\left ( u, v \right ) & \mapsto \left ( \begin{array}{c}
u _ 1 - \pi \\
v _ 2 + 1
\end{array} \right )
\end{aligned} \right .
```

"""

# ╔═╡ 97952745-6959-461b-91b4-9fa75769a0c4
function example_bc!(res, left, right)
	res[1] = left[1] - π
	res[2] = right[2] + 1
	nothing
end

# ╔═╡ edbe2e42-29fb-4522-ba93-39e23f72d657
function bc!(res, left, right)
	res[1] = left[1]
	res[2] = left[2]
	res[3] = right[2] - 1
	nothing
end

# ╔═╡ d3e55c3d-b6b2-4e37-9206-1b83ce3bae0e
function shooting(scheme!, f, τ, s, bc!, y₀)
	y = getproperty(
		nlsolve(y₀) do res, y
			_, Y = cauchy(scheme!, f, τ, s, y)

			bc!(res, y, last(Y))
		end,
		:zero
	)
	cauchy(scheme!, f, τ, s, y)
end

# ╔═╡ 0918d061-0b61-4d6f-92c6-3a3c3c059320
shooting(midpoint!, source, 0.01, 10.0, bc!, [0.0; 0.0; 1.0])

# ╔═╡ 976d6502-c1d0-4232-a391-d6d3d2165d73
begin
	local T, Y = shooting(midpoint!, source, 0.01, 10.0, bc!, [0.0; 0.0; 1.0])
	local fig = plot()
	scatter!(fig, (x -> getindex(x, 2)).(Y), T, label = "f'")
end

# ╔═╡ Cell order:
# ╟─dc4e617c-78d2-4c38-b03d-971a7a82c790
# ╠═82c64cc3-81a7-45d8-87b4-81f28b8c0a9d
# ╠═cd52be0a-6128-4e25-ab0c-885620fefcbd
# ╠═ebf51cb6-dce4-457d-b94e-cd3c382905fe
# ╠═8c7a1da1-d048-4724-9375-31adc372dbb1
# ╠═0d54c141-0114-457d-8332-1551cb8984ec
# ╠═ea19915a-76d5-4ad1-97fd-e096aa38aa90
# ╠═1624bbe7-ebd5-415e-9a76-dbbea721124d
# ╠═acd1c234-f035-4f04-bf1a-2ec357657117
# ╟─983d360a-ac0c-4e4b-a4d6-55f3d80f43b5
# ╠═30bf3035-ca50-4086-ab2e-888aed6eadf7
# ╠═b6cd4932-c551-406e-8ba6-ee93f9c436c8
# ╟─44b3bef6-f820-492f-8d01-6792d7495759
# ╠═97952745-6959-461b-91b4-9fa75769a0c4
# ╠═edbe2e42-29fb-4522-ba93-39e23f72d657
# ╠═d3e55c3d-b6b2-4e37-9206-1b83ce3bae0e
# ╠═0918d061-0b61-4d6f-92c6-3a3c3c059320
# ╠═976d6502-c1d0-4232-a391-d6d3d2165d73
