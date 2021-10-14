### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ 1648f2aa-11a9-4d2b-8816-4d4a43f046f9
using LinearAlgebra, Plots

# ╔═╡ 71420523-711d-4811-bb75-abc17ceb0875
md"""
Le cas Dirichlet pur :
```math
y_i'' \simeq \frac{y_{i - 1} - 2 y_i + y_{i + 1}}{h ^ 2}.
```

On commence par définir le pas en espace :
```math
h = \frac{L}{N + 1}.
```

"""

# ╔═╡ 6067d7e0-fbbe-48b9-90c9-f5aacbacac90
# solution exacte avec les conditions de Dirichlet (0 à gauche, 1 à droite)
solution(x, l = π) = x / l

# ╔═╡ 6e017c65-4169-4294-a308-ad758cc1e6a5
# pas en espace
spacing(n, l = π) = l / (n + 1)

# ╔═╡ b14a53a6-cf3e-11eb-127e-e19ef96d3422
# assemblage de A
function laplacian(n)
	h = spacing(n)
	l = ones(n - 1) / h ^ 2
	d = -2ones(n) / h ^ 2
	u = ones(n - 1) / h ^ 2
	Tridiagonal(l, d, u)
end

# ╔═╡ 78ccc288-8e1a-492a-8034-d6692171e7bb
# second membre (b)
function rhs(n, α = 0, β = 1)
	h = spacing(n)
	b = zeros(n)
	b[begin] = -α / h ^ 2
	b[end] = -β / h ^ 2
	b
end

# ╔═╡ 8f48dc09-6012-4a31-9ab3-62abc242959b
md"""
!!! note "Résolution du système linéaire"

	On utilise l'opérateur `\` pour inverser le système
	```math
	A x = b
	```
	comme suite :
	```julia
	x = A \ b
	```

"""

# ╔═╡ d3bacb41-5fba-4ae1-b80a-8c8264d63d0f
# résolution du système
function steady(n)
	A = laplacian(n)
	b = rhs(n)
	A \ b
end

# ╔═╡ b8714ec3-a781-45b8-9e75-ebdf5ff62f18
md"""
```math
x_i = h i = \frac{L i}{N + 1}, \quad i = 1\ldots N.
```

"""

# ╔═╡ 9e2c981e-9e7e-49bc-b988-c7a78c0016fc
function abscissa(n)
	h = spacing(n)
	x = [i * h for i in 1:n]
end

# ╔═╡ 2ecc05d0-8227-4222-992e-2fe892d07257
# FDM = Finite Difference Method
begin
	local n = 6
	local x = abscissa(n)
	local y = steady(n)
	local fig = plot(xlim = (0, π), ylim = (0, 1))
	plot!(fig, solution, label = "exacte")
	scatter!(fig, x, y, label = "FDM")
end

# ╔═╡ Cell order:
# ╠═1648f2aa-11a9-4d2b-8816-4d4a43f046f9
# ╟─71420523-711d-4811-bb75-abc17ceb0875
# ╠═6067d7e0-fbbe-48b9-90c9-f5aacbacac90
# ╠═6e017c65-4169-4294-a308-ad758cc1e6a5
# ╠═b14a53a6-cf3e-11eb-127e-e19ef96d3422
# ╠═78ccc288-8e1a-492a-8034-d6692171e7bb
# ╟─8f48dc09-6012-4a31-9ab3-62abc242959b
# ╠═d3bacb41-5fba-4ae1-b80a-8c8264d63d0f
# ╟─b8714ec3-a781-45b8-9e75-ebdf5ff62f18
# ╠═9e2c981e-9e7e-49bc-b988-c7a78c0016fc
# ╠═2ecc05d0-8227-4222-992e-2fe892d07257
