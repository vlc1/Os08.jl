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

# ╔═╡ 0918d061-0b61-4d6f-92c6-3a3c3c059320
shooting(midpoint!, source, 0.01, 1.0, bc!, [1.0; 0.0])

# ╔═╡ 976d6502-c1d0-4232-a391-d6d3d2165d73
begin
	local T, Y = shooting(midpoint!, source, 0.01, 1.0, bc!, [1.0; 0.0])
	local fig = plot()
	scatter!(fig, T, first.(Y), label = "T")
	scatter!(fig, T, last.(Y), label = "T'")
end

# ╔═╡ 597f7932-5df2-4a8d-8121-7ea3abeb619a
md"""
# Fonte d'un glaçon

On considère la fonte d'un glaçon, pour lequel les profils de température (dans le référentiel du front) sont sujets à l'équation
```math
T'' \left ( x \right ) = 0
```
dans les domaines solide (``\left ] 0, 1 / 2 \right [``) et liquide (``\left ] 1 / 2, 1 \right [``).

On suppose que les températures connues aux deux extrémités
```math
\left \{ \begin{aligned}
T \left ( 0 \right ) & = 0, \\
T \left ( 1 \right ) & = 1.
\end{aligned} \right .
```

À l'interface (``x = 1 / 2``), la température et le flux conductif de chaleur sont supposés continus
```math
\lim _ {\epsilon \to 0} \; T \left ( \frac{1}{2} - \epsilon \right ) = \lim _ {\epsilon \to 0} \; T \left ( \frac{1}{2} + \epsilon \right )
```
et
```math
\lim _ {\epsilon \to 0} \; \kappa _ 1 T' \left ( \frac{1}{2} - \epsilon \right ) = \lim _ {\epsilon \to 0} \; \kappa _ 2 T' \left ( \frac{1}{2} + \epsilon \right ).
```
On prendra pour valeurs ``\kappa _ 1 = 2`` et ``\kappa _ 2 = 2``. Cette EDO appartient à la catégorie des *multipoint boundary value problems*.

1. Réécrire ce système sous la forme d'un *two-point boundary value problem*.

Il suffit de différencier les deux champs de température, à savoir
```math
T _ 1 \colon \left \vert \begin{aligned}
\left ] 0, 1 / 2 \right [ & \to \mathbb{R}, \\
\xi & \mapsto T \left ( \xi \right )
\end{aligned} \right .
```
et
```math
T _ 2 \colon \left \vert \begin{aligned}
\left ] 0, 1 / 2 \right [ & \to \mathbb{R}, \\
\xi & \mapsto T \left ( \xi + 1 / 2 \right )
\end{aligned} \right .
```

Le problème, qui doit maintenant être résolu sur l'intervalle ``\left ] 0, 1 / 2 \right [``, s'écrit alors
```math
\left \{ \begin{aligned}
T'' _ 1 \left ( \xi \right ) & = 0, \\
T'' _ 2\left ( \xi \right ) & = 0,
\end{aligned} \right . \quad \xi \in \left ] 0, 1 / 2 \right [
```
avec les conditions aux limites
```math
\left \{ \begin{aligned}
T _ 1 \left ( 0 \right ) & = 0, \\
T _ 1 \left ( 1 / 2 \right ) & = T _ 2 \left ( 0 \right ), \\
\kappa _ 1 T' _ 1 \left ( 1 / 2 \right ) & = \kappa _ 2 T' _ 2 \left ( 0 \right ), \\
T _ 2 \left ( 1 / 2 \right ) & = 1.
\end{aligned} \right .
```

2. Réécrire ces deux EDO d'ordre 2 sous la forme d'un système d'EDO d'ordre 1.

Il suffit d'introduire
```math
\left \{ \begin{aligned}
u _ 1 & = T _ 1, \\
u _ 2 & = T' _ 1, \\
u _ 3 & = T _ 2, \\
u _ 4 & = T' _ 2.
\end{aligned} \right .
```
Le système d'EDO s'écrit alors
```math
\mathbf{u}' \left ( \xi \right ) = \left ( \begin{array}{c}
u _ 2 \left ( \xi \right ) \\
0 \\
u _ 4 \left ( \xi \right ) \\
0
\end{array} \right ), \quad \xi \in \left ] 0, 1 / 2 \right [.
```

Pour les conditions aux limites (`bc!`)
```math
h \colon \left \vert \begin{aligned}
\mathbb{R} ^ 4 \times \mathbb{R} ^ 4 & \to \mathbb{R} ^ 4, \\
\left ( \mathbf{l}, \mathbf{r} \right ) & \mapsto \left ( l _ 1, r _ 1 - l _ 3, r _ 2 - 2 l _ 4, r _ 3 - 1 \right )
\end{aligned} \right .
```

"""

# ╔═╡ 0d7ed618-6d9a-4569-94f7-2e17884a5a18
function ice_source(t, y)
	x = similar(y)
	x[1] = y[2]
	x[2] = 0
	x[3] = y[4]
	x[4] = 0
	return x
end

# ╔═╡ 6ee56e07-e9da-4029-9499-3895d7ca387d
function ice_bc!(res, left, right)
	res[1] = left[1]
	res[2] = right[1] - left[3]
	res[3] = right[2] - 2left[4]
	res[4] = right[3] - 1
	nothing
end

# ╔═╡ 7206b5cd-e696-47e7-9ecd-62b7f03a955a
begin
	local T, Y = shooting(midpoint!, ice_source, 0.01, 0.5, ice_bc!, [0.0; 1.0; 1.0; 0.0])
	local fig = plot(xlim = (0, 1))
	scatter!(fig, T, first.(Y), label = "T₁")
	scatter!(fig, T .+ 0.5, (x -> getindex(x, 3)).(Y), label = "T₂")
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
# ╠═30bf3035-ca50-4086-ab2e-888aed6eadf7
# ╟─9f840072-38dc-4b57-9556-aaa02b90fa4a
# ╠═b6cd4932-c551-406e-8ba6-ee93f9c436c8
# ╟─c8c08cd3-f216-45a0-acee-75d2be6dded6
# ╠═edbe2e42-29fb-4522-ba93-39e23f72d657
# ╠═d3e55c3d-b6b2-4e37-9206-1b83ce3bae0e
# ╠═0918d061-0b61-4d6f-92c6-3a3c3c059320
# ╠═976d6502-c1d0-4232-a391-d6d3d2165d73
# ╟─597f7932-5df2-4a8d-8121-7ea3abeb619a
# ╠═0d7ed618-6d9a-4569-94f7-2e17884a5a18
# ╠═6ee56e07-e9da-4029-9499-3895d7ca387d
# ╠═7206b5cd-e696-47e7-9ecd-62b7f03a955a
