### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ b2123093-fc72-4cc4-b665-9452b6317b81
using LinearAlgebra, Plots, NLsolve

# ╔═╡ 40a5c1ba-2cfb-11ec-0467-b15d1340a3b4
md"""
# Résolution de l'équation de la chaleur

```math
\frac{\partial T}{\partial t} = \frac{\partial ^ 2 T}{\partial x ^ 2}, \quad t > 0, \quad 0 < x < 1
```
avec pour condition initiale (`initial`)
```math
T \left ( 0, x \right ) = T _ 0 \left ( x \right )
```
et pour conditions aux limites
```math
\left \{ \begin{aligned}
T \left ( t, 0 \right ) & = f \left ( t \right ), \\
T \left ( t, 1 \right ) & = g \left ( t \right )
\end{aligned} \right .
```
implémentées par les fonctions `left` et `right` ci-dessous.

"""

# ╔═╡ 893dc203-f562-4c91-8e9f-2caa91537d76
function initial(x)
	sin(π * x / 2)
end

# ╔═╡ 2ff5164f-3484-4ca2-a4bf-8db5b93dcafb
function left(t)
	zero(t)
end

# ╔═╡ 65472157-e7c4-41ea-8465-900e9e8abe85
function right(t)
	one(t)
end

# ╔═╡ e72f85b4-c09a-44de-8863-339d456550e2
md"""
# Création du maillage et de la condition initiale

"""

# ╔═╡ 80090344-de8f-4b45-a20d-937786f051af
spacing(n, l = 1) = l / (n + 1)

# ╔═╡ ec47978f-83e3-4936-a940-5a928506b09b
function abscissa(n)
	h = spacing(n)
	x = [i * h for i in 1:n]
end

# ╔═╡ a4688dc7-5260-419e-8f79-57a7fe90c91f
n = 10

# ╔═╡ 059078df-32e3-4336-b99a-e3b2bd959b3b
x = abscissa(n)

# ╔═╡ 6c935a4e-b877-4148-a638-a3f539c7334f
u₀ = initial.(x)

# ╔═╡ e9e5d4a9-0b4b-4e1f-af52-4312a04d6e0a
scatter(x, u₀, label = "T(0, x)")

# ╔═╡ 06aeb061-10e3-4579-8c44-03ee48012573
function laplacian(n)
	h = spacing(n)
	l = ones(n - 1) / h ^ 2
	d = -2ones(n) / h ^ 2
	u = ones(n - 1) / h ^ 2
	Tridiagonal(l, d, u)
end

# ╔═╡ 9d0cf0d9-ddc8-4872-b9b9-4fe39b8fed2a
function rhs(n, α, β)
	h = spacing(n)
	b = zeros(n)
	b[begin] = α / h ^ 2
	b[end] = β / h ^ 2
	b
end

# ╔═╡ ad228c21-8c77-457f-81de-a84c6d3d3a3d
function heat(t, y, n = n, f = left, g = right)
	A = laplacian(n)
	b = rhs(n, f(t), g(t))
	A * y + b
end

# ╔═╡ 1fb6412c-e7a4-4669-b99d-c93e7689d0d2
midpoint!(res, x, y, τ, f, t) = res .= x - y - τ * f(t + τ / 2, (x + y) / 2)

# ╔═╡ fe174bf3-1ec4-4f22-8f64-6d086bfbf66f
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

# ╔═╡ 83ed8bde-591d-45f3-af16-d433a51c696f
T, Y = cauchy(midpoint!, heat, 0.01, 1.0, u₀)

# ╔═╡ 08e6fbe7-49bd-4a60-9b0f-567b02f8a25c
begin
	local fig = scatter(xlim = (0, 1), ylim = (0, 1))
	local x = abscissa(n)
	scatter!(fig, x, last(Y))
end

# ╔═╡ Cell order:
# ╠═b2123093-fc72-4cc4-b665-9452b6317b81
# ╟─40a5c1ba-2cfb-11ec-0467-b15d1340a3b4
# ╠═893dc203-f562-4c91-8e9f-2caa91537d76
# ╠═2ff5164f-3484-4ca2-a4bf-8db5b93dcafb
# ╠═65472157-e7c4-41ea-8465-900e9e8abe85
# ╟─e72f85b4-c09a-44de-8863-339d456550e2
# ╠═80090344-de8f-4b45-a20d-937786f051af
# ╠═ec47978f-83e3-4936-a940-5a928506b09b
# ╠═a4688dc7-5260-419e-8f79-57a7fe90c91f
# ╠═059078df-32e3-4336-b99a-e3b2bd959b3b
# ╠═6c935a4e-b877-4148-a638-a3f539c7334f
# ╠═e9e5d4a9-0b4b-4e1f-af52-4312a04d6e0a
# ╠═06aeb061-10e3-4579-8c44-03ee48012573
# ╠═9d0cf0d9-ddc8-4872-b9b9-4fe39b8fed2a
# ╠═ad228c21-8c77-457f-81de-a84c6d3d3a3d
# ╠═1fb6412c-e7a4-4669-b99d-c93e7689d0d2
# ╠═fe174bf3-1ec4-4f22-8f64-6d086bfbf66f
# ╠═83ed8bde-591d-45f3-af16-d433a51c696f
# ╠═08e6fbe7-49bd-4a60-9b0f-567b02f8a25c
