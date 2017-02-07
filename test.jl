A = rand(3, 3)

s = svdvals(A)

U, S, V = svd(A)

λ = eigvals(A)

using Plots; gr()
heatmap(-A)
