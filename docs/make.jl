using RForest
using Documenter

DocMeta.setdocmeta!(RForest, :DocTestSetup, :(using RForest); recursive=true)

makedocs(;
    modules=[RForest],
    authors="JJ Garzella <jjgarzella@gmail.com> and contributors",
    sitename="RForest.jl",
    format=Documenter.HTML(;
        canonical="https://jjgarzella.github.io/RForest.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jjgarzella/RForest.jl",
    devbranch="main",
)
