### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ 82c64cc3-81a7-45d8-87b4-81f28b8c0a9d
using LinearAlgebra, NLsolve, Plots, LsqFit

# ╔═╡ dc4e617c-78d2-4c38-b03d-971a7a82c790
md"""
# Prérequis

!!! warning "Remarque"

	Les cellules suivantes ont été copiées du TP précédent. Merci de ne pas les modifier.

	À la différence des deux derniers TP, celui-ci ne sera pas évalué mais réalisé en classe ensemble.

"""

# ╔═╡ dfe00637-6820-4a35-bea5-1fc9ebdee50b
linear(t, y) = -y

# ╔═╡ 19b2ba7b-833b-4f8d-8369-c944753d11e9
solution(t, y = ones(1)) = exp(-t) * y

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

# ╔═╡ 12a27568-208a-4ab6-bdc6-f0c01d07e02c
function error(scheme!, τ, s)
	T, num = cauchy(scheme!, linear, τ, s, ones(1))
	exact = solution.(T)
	norm(last(num) - last(exact))
end

# ╔═╡ 26f95768-17bb-4ddf-b1b5-bafacb73aea0
md"""
# Domaine de stabilité

Les domaines de stabilité de plusieurs schémas, notamment celui de la méthode explicite d'Euler
```math
\left \{ z \in \mathbb{C} \left  / \left \vert 1 + z \right \vert \le 1 \right . \right \},
```
ont été présentés en cours.

On peut montrer de plus que pourles schémas explicites de Runge-Kutta d'ordre ``p``, la raison ``\sigma`` peut s'écrire sous la forme d'un développement limité de ``\exp`` à l'ordre ``p``,
```math
\sigma \colon z \mapsto \sum_{n = 0} ^ p \frac{z ^ n}{n !}.
```
La fonction `ratio` ci-dessous implémente cette fonction des arguments ``x`` et ``y``, à savoir les parties réelle et imaginaire de ``z = x + \imath y``.

"""

# ╔═╡ 3a695cb5-40e0-4b73-9e8b-2d2f0928e74c
for i in 1:4
	eval(:(order(::typeof($(Symbol("rk$(i)!")))) = $(i)))
end

# ╔═╡ 81ec5798-c75b-11eb-3e50-47f7b9194ece
function ratio(rk!, x, y)
	z = x + y * im
	t = zero(z)
	for j in 0:order(rk!)
		t += z ^ j / factorial(j)
	end
	abs(t)
end

# ╔═╡ 666d3cf9-4764-46b1-ba12-84c23f849d84
begin
	local x, y = -3:0.1:1, -3:0.1:3
	local fig = plot(xlim = (x[begin], x[end]), ylim = (y[begin], y[end]))
	for (i, rk!) in enumerate([rk1!, rk2!, rk3!, rk4!])
		contour!(fig,
			x, y,
			(x, y) -> ratio(rk!, x, y),
			levels = [1.],
			aspect_ratio = :equal,
			lw = 2,
			color = i
		)
	end
	fig
end

# ╔═╡ 64c9a088-4e3e-4a7e-a73f-e2501cf6ba5d
md"""
1. Dans le cas du modèle linéaire (``f \colon \left ( t, y \right ) \mapsto \lambda y``) où ``\lambda \in \mathbb{R} ^ -``, la fonction `getmax` implémentée ci-dessous permet de calculer le pas de temps au delà duquel les méthodes explicites de RK1, RK2, RK3 et RK4 deviennent instables. Calculer ces valeurs et remplir le tableau ci-dessous.

|               | RK1 | RK2 | RK3 | RK4 |
|:-------------:|:---:|:---:|:---:|:---:|
| ``\tau_\max`` |     |     |     |     |

"""

# ╔═╡ 1bc9edca-2ed3-444b-a448-a874c5a12ed2
function getmax(rk!, x₀ = [-2.0])
	res = similar(x₀)
	first(
		getproperty(
		nlsolve(x₀) do res, x
			res[begin] = ratio(rk!, x[begin], zero(eltype(x))) - one(eltype(x))
		end,
		:zero)
	)
end

# ╔═╡ 5505d3e2-6978-472e-8fa5-6c8483f236ba
xmax = getmax(rk3!)

# ╔═╡ 7824c63f-44bb-432f-9c5f-04b336348250
md"""
2. Vérifier qu'au delà de ces valeurs, chaque méthode devient instable. On pourra s'inspirer du code suivant.

"""

# ╔═╡ 42426ddb-1a42-454b-ba3a-52f792170067
begin
	local xmax = getmax(rk4!)
	local s = 20.
	local T, Y = cauchy(rk4!, linear, -xmax + 0.1, 10, ones(1))
	local Z = solution.(T)
	local fig = plot(xlim = (0, s))
	plot!(fig, first∘solution, label = "Exacte")
	scatter!(fig, T, first.(Y), label = "Numérique")
end

# ╔═╡ cf81908f-beab-4bbe-a092-e317557825f7
md"""
# Ordre de convergence

On rappelle que l'erreur d'un schéma à l'instant ``s = t _ N`` (où ``N = s / \tau``), définie par
```math
\epsilon = y_N - y \left ( t_N \right ),
```
peut s'écrire
```math
\epsilon = C \tau ^ p
```
où ``p`` est appelé l'**ordre** de la méthode. En composant par la fonction ``\ln`` on obtient alors que ``\ln \epsilon`` est une fonction affine de ``\ln \tau`` :
```math
y = \ln \epsilon = p \ln \tau + C
```
dont le coefficient directeur ``p`` est l'ordre.

Le code suivante réutilise la fonction `error` du TP2, et l'applique à plusieurs schémas et pas de temps avant de la visualiser en échelle logarithmique.

3. Commenter l'allure des courbes.

"""

# ╔═╡ de51eb84-dfaa-412d-9a76-7d1f5a6cecdb
begin
	local s = 2.0
	local schemes! = [explicit!, implicit!, midpoint!, rk2!, rk3!]
	τs = [1 / 2 ^ i for i in 7:-1:3]
	ϵs = Dict(scheme! => Float64[] for scheme! in schemes!)
	for scheme! in schemes!
		for τ in τs
			append!(ϵs[scheme!], error(scheme!, τ, s))
		end
	end

	local fig = plot(scale = :log, legend=:bottomright)
	for key in keys(ϵs)
		scatter!(fig, τs, ϵs[key], label = "$(key)")
	end
	fig
end

# ╔═╡ 86061c1c-4df9-45ea-ad48-c24ca151f1ca
md"""
On peut aussi utiliser la méthode des moindres carrés pour estimer l'ordre de ces schémas (package `LsqFit`).

4. Utiliser la fonction `ordernum` implémentée ci-dessous pour estimer l'ordre des méthodes `explicit!`, `implicit!`, `midpoint!`, `rk2!` et `rk3!`.

"""

# ╔═╡ 8fae3217-945b-4bde-916d-b396346122ff
model(x, p) = p[1] * x .^ p[2]

# ╔═╡ a4726400-8ad0-45d3-b201-f333942dd425
ordernum(τs, ϵs, p₀ = [1.0; 2.0]) =
	last(
		getproperty(
			curve_fit(model, τs, ϵs, [0.5; 0.5]),
			:param
		)
	)

# ╔═╡ fa291891-0703-4b75-99a7-9113e2a81b83
ordernum(τs, ϵs[rk3!])

# ╔═╡ 9f840072-38dc-4b57-9556-aaa02b90fa4a
md"""
# Conduction stationnaire dans une barre

On se propose de résoudre l'équation
```math
y'' \left ( x \right ) = \sin \left ( \pi x \right )
```
avec les conditions
```math
\left \{ \begin{aligned}
y \left ( 0 \right ) & = 1, \\
y' \left ( 1 \right ) & = 0.
\end{aligned} \right .
```
par la méthode de tir. On commence par réécrire l'équation scalaire ci-dessus sous la forme d'un système de deux EDO couplées :
```math
\left \{ \begin{aligned}
u' \left ( t \right ) & = v \left ( t \right ), \\
v' \left ( t \right ) & = \sin \left ( \pi t \right ).
\end{aligned}
\right .
```

Ce modèle est implémenté par la fonction `source` ci-dessous.

"""

# ╔═╡ b6cd4932-c551-406e-8ba6-ee93f9c436c8
function source(t, y)
	x = similar(y)
	x[1] = y[2]
	x[2] = sin(π * t)
	return x
end

# ╔═╡ c8c08cd3-f216-45a0-acee-75d2be6dded6
md"""
La spécification des conditions aux limites est plus complex. On se propose ici de considérer trois types de conditions aux limites :

- Dirichlet, tel que ``T \left ( c \right ) = 1`` ;
- Neumann, tel que ``T' \left ( c \right ) = 0`` (homogène) ;
- Robin, tel que ``T \left ( c \right ) - \pi T' \left ( c \right ) = 0`` (homogène).

Dans ces équations, ``c \in \left \{ 0, 1 \right \}`` (les bornes du domaine).

"""

# ╔═╡ edbe2e42-29fb-4522-ba93-39e23f72d657
function bc!(res, left, right)
	res[1] = left[1] - 1.0
	res[2] = right[2] - 0.0
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

# ╔═╡ 976d6502-c1d0-4232-a391-d6d3d2165d73
begin
	local T, Y = shooting(midpoint!, source, 0.01, 1.0, bc!, [1.0; 0.0])
	local fig = plot()
	scatter!(fig, T, first.(Y), label = "T")
	scatter!(fig, T, last.(Y), label = "T'")
end

# ╔═╡ c8ca6175-f88d-4b00-9999-0fd61a7456f4
md"""
**Prolongation possible** : interpolation d'Hermite.

"""

# ╔═╡ Cell order:
# ╟─dc4e617c-78d2-4c38-b03d-971a7a82c790
# ╠═82c64cc3-81a7-45d8-87b4-81f28b8c0a9d
# ╠═dfe00637-6820-4a35-bea5-1fc9ebdee50b
# ╠═19b2ba7b-833b-4f8d-8369-c944753d11e9
# ╠═12a27568-208a-4ab6-bdc6-f0c01d07e02c
# ╠═cd52be0a-6128-4e25-ab0c-885620fefcbd
# ╠═ebf51cb6-dce4-457d-b94e-cd3c382905fe
# ╠═8c7a1da1-d048-4724-9375-31adc372dbb1
# ╠═0d54c141-0114-457d-8332-1551cb8984ec
# ╠═ea19915a-76d5-4ad1-97fd-e096aa38aa90
# ╠═1624bbe7-ebd5-415e-9a76-dbbea721124d
# ╠═acd1c234-f035-4f04-bf1a-2ec357657117
# ╠═30bf3035-ca50-4086-ab2e-888aed6eadf7
# ╟─26f95768-17bb-4ddf-b1b5-bafacb73aea0
# ╠═3a695cb5-40e0-4b73-9e8b-2d2f0928e74c
# ╠═81ec5798-c75b-11eb-3e50-47f7b9194ece
# ╠═666d3cf9-4764-46b1-ba12-84c23f849d84
# ╟─64c9a088-4e3e-4a7e-a73f-e2501cf6ba5d
# ╠═1bc9edca-2ed3-444b-a448-a874c5a12ed2
# ╠═5505d3e2-6978-472e-8fa5-6c8483f236ba
# ╟─7824c63f-44bb-432f-9c5f-04b336348250
# ╠═42426ddb-1a42-454b-ba3a-52f792170067
# ╟─cf81908f-beab-4bbe-a092-e317557825f7
# ╠═de51eb84-dfaa-412d-9a76-7d1f5a6cecdb
# ╟─86061c1c-4df9-45ea-ad48-c24ca151f1ca
# ╠═8fae3217-945b-4bde-916d-b396346122ff
# ╠═a4726400-8ad0-45d3-b201-f333942dd425
# ╠═fa291891-0703-4b75-99a7-9113e2a81b83
# ╟─9f840072-38dc-4b57-9556-aaa02b90fa4a
# ╠═b6cd4932-c551-406e-8ba6-ee93f9c436c8
# ╟─c8c08cd3-f216-45a0-acee-75d2be6dded6
# ╠═edbe2e42-29fb-4522-ba93-39e23f72d657
# ╠═d3e55c3d-b6b2-4e37-9206-1b83ce3bae0e
# ╠═976d6502-c1d0-4232-a391-d6d3d2165d73
# ╟─c8ca6175-f88d-4b00-9999-0fd61a7456f4
