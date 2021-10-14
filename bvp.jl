### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ 8b7b1636-bdc5-4293-92e1-bf3019dbee85
using LinearAlgebra, NLsolve, Plots

# ╔═╡ 07cbb1bf-7943-413c-8e3b-fb823e7fd3c8
md"""
# Prérequis

!!! warning "Remarque"

	Les cellules suivantes ont été copiées des TPs précédents. Merci de ne pas les modifier.

"""

# ╔═╡ bfa6143a-1b2f-4eda-a647-3ef875d84f1d
explicit!(res, x, y, τ, f, t) = res .= x - y - τ * f(t, y)

# ╔═╡ 9daa0844-f897-46d7-b1f9-d03743b35001
implicit!(res, x, y, τ, f, t) = res .= x - y - τ * f(t + τ, x)

# ╔═╡ fad7b304-8bf7-4dfc-ac4c-608264f78cc3
midpoint!(res, x, y, τ, f, t) = res .= x - y - τ * f(t + τ / 2, (x + y) / 2)

# ╔═╡ 0cafcd96-77ca-4688-b0c9-ec608f3cc51f
function rk2!(res, x, y, τ, f, t)
	k₁ = f(t, y)
	k₂ = f(t + τ / 2, y + τ * k₁ / 2)
	res .= x - y - τ * k₂
end

# ╔═╡ 89ee0e75-0677-4143-a5a5-00a2e70e31c1
function rk3!(res, x, y, τ, f, t)
	k₁ = f(t, y)
	k₂ = f(t + τ / 3, y + τ * k₁ / 3)
	k₃ = f(t + 2τ / 3, y + 2τ * k₂ / 3)
	res .= x - y - τ * (k₁ + 3k₃) / 4
end

# ╔═╡ 3505286b-a72e-4b45-8696-d69a681aaf77
function rk4!(res, x, y, τ, f, t)
	k₁ = f(t, y)
	k₂ = f(t + τ / 2, y + τ * k₁ / 2)
	k₃ = f(t + τ / 2, y + τ * k₂ / 2)
	k₄ = f(t + τ, y + τ * k₃)
	res .= x - y - τ * (k₁ + 2k₂ + 2k₃ + k₄) / 6
end

# ╔═╡ 6abc9579-e62e-44eb-8ce2-d7f96a78a1d0
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

# ╔═╡ 46959773-4505-410e-a25b-4467040a8606
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

# ╔═╡ 81b7c1fa-cecb-11eb-0162-852167b5ca8e
md"""
# Conduction en régime stationnaire

On se propose de résoudre le problème aux limites suivant :
```math
y'' \left ( x \right ) + \frac{1}{4} y \left ( x \right ) = 0, \quad x \in \left ] 0, \pi \right [,
```
avec les conditions aux limites
```math
\left \{ \begin{aligned}
y \left ( 0 \right ) & = 0, \\
y \left ( \pi \right ) & = 1.
\end{aligned} \right .
```

1. Résoudre analytiquement ce problème. Le réécrire sous la forme d'un système de deux équations différentielles ordinaires couplées (``u``, ``v``).

!!! note "Remarque"

	Ne pas omettre les conditions aux limites. On cherchera la solution sous la forme
	```math
	y \colon x \mapsto A \sin \left ( \omega x \right ) + B \cos \left ( \omega x \right )
	```
	où ``\omega`` est à déterminer.

2. Implémenter la solution analytique dans la fonction `solution` ci-dessous.

"""

# ╔═╡ 3e7afecc-31eb-4430-8ac5-08fbe74db93d
# Q2
solution(x) = zeros(2)

# ╔═╡ 7ba07c59-eafe-49c3-af7e-b73c2d93a3bc
md"""
3. En vous inspirant du cahier du TP4, implémenter les fonctions `source` et `bc!` qui correspondent au système obtenu à la question 1 et seront passées à la fonction `shooting`.

"""

# ╔═╡ 5e451606-839b-4ae4-8492-736922b800ed
# Q3a
function source(t, y)
	x = similar(y)
	x[1] = 0
	x[2] = 0
	return x
end

# ╔═╡ cbd543b0-183c-464d-b365-e0dfc783d92c
# Q3b
function bc!(res, left, right)
	res[1] = 0
	res[2] = 0
	nothing
end

# ╔═╡ aba85635-698e-426e-9b51-70c257bb3690
md"""
4. Résoudre le problème par la méthode de tir avec les paramètres de votre choix.
5. En modifiant le cahier réalisé en TD, résoudre l'équation par la méthode des différences finies.
6. On définit l'erreur par
```math
\epsilon_N = \sqrt{\sum_{n = 1} ^ N \frac{\left [ y_n - y \left ( x_n \right ) \right ] ^ 2}{N}}
```
où ``N`` dénote le nombre de points **à l'intérieur de l'intervalle** ``\left ] 0, \pi \right [`` auxquelles les solutions numériques (par les méthodes de tir ou des différences finies) sont calculées, et ``\left ( x _n \right )`` les abscisses correspondantes. Remplir le tableau ci-dessous.

|         ``N``            | ``16`` | ``32`` | ``64`` | ``128`` |
|:------------------------:|:------:|:------:|:------:|:-------:|
| *Shooting* + `implicit!` |        |        |        |         |
| *Shooting* + `midpoint!` |        |        |        |         |
| *Shooting* + `rk4!`      |        |        |        |         |
| Différences finies       |        |        |        |         |

"""

# ╔═╡ 3f980e32-e2c5-4679-a6a8-0caad4249817
md"""
# Conduction en régime instationnaire

On se propose de résoudre l'équation
```math
\frac{\partial y}{\partial t} = \frac{\partial ^ 2 y}{\partial x ^ 2}, \quad t > 0, \quad 0 < x < \pi,
```
soumis à la condition initiale
```math
y \left (0, x \right ) = 0, \quad 0 < x < \pi,
```
et aux conditions aux limites
```math
\left \{ \begin{aligned}
y \left ( t, 0 \right ) & = 0, \\
y \left ( t, \pi \right ) & = 1,
\end{aligned} \right .\quad t > 0.
```

7. En vous inspirant du cahier réalisé en TD, compléter la fonction `unsteady` ci-dessous pour qu'elle corresponde à la discrétisation de la dérivée seconde
```math
\frac{\partial ^ 2 y}{\partial x ^ 2}.
```
par la méthode des différences finies.

!!! note "Remarque"

	Ne pas oublier les conditions aux limites.

"""

# ╔═╡ 7ecac97a-f93f-4bd7-a59d-738d7de2d00e
# Q7
function unsteady(t, y)
	x = zero(y)
    n = length(y)
	return x
end

# ╔═╡ 4997c413-2ac2-4913-a5e3-806ab0c93605
md"""
8. Résoudre le problème de Cauchy qui découle de la discrétisation spatiale de la question 8 par la méthode de votre choix. Visualiser la solution à plusieurs instants compris entre ``0 < t < 1``, puis en plusieurs points ``0 < x < \pi`` en fonction du temps sur le même intervalle (``\left [ 0, 1 \right ]``).

"""

# ╔═╡ b9497f6e-29ea-4efb-a2e2-73caa5b51511
y₀ = zeros(3)

# ╔═╡ Cell order:
# ╟─07cbb1bf-7943-413c-8e3b-fb823e7fd3c8
# ╠═8b7b1636-bdc5-4293-92e1-bf3019dbee85
# ╠═bfa6143a-1b2f-4eda-a647-3ef875d84f1d
# ╠═9daa0844-f897-46d7-b1f9-d03743b35001
# ╠═fad7b304-8bf7-4dfc-ac4c-608264f78cc3
# ╠═0cafcd96-77ca-4688-b0c9-ec608f3cc51f
# ╠═89ee0e75-0677-4143-a5a5-00a2e70e31c1
# ╠═3505286b-a72e-4b45-8696-d69a681aaf77
# ╠═6abc9579-e62e-44eb-8ce2-d7f96a78a1d0
# ╠═46959773-4505-410e-a25b-4467040a8606
# ╟─81b7c1fa-cecb-11eb-0162-852167b5ca8e
# ╠═3e7afecc-31eb-4430-8ac5-08fbe74db93d
# ╟─7ba07c59-eafe-49c3-af7e-b73c2d93a3bc
# ╠═5e451606-839b-4ae4-8492-736922b800ed
# ╠═cbd543b0-183c-464d-b365-e0dfc783d92c
# ╟─aba85635-698e-426e-9b51-70c257bb3690
# ╟─3f980e32-e2c5-4679-a6a8-0caad4249817
# ╠═7ecac97a-f93f-4bd7-a59d-738d7de2d00e
# ╟─4997c413-2ac2-4913-a5e3-806ab0c93605
# ╠═b9497f6e-29ea-4efb-a2e2-73caa5b51511
