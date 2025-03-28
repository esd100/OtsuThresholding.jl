using OtsuThresholding
using Documenter

# Automatically generate doctrings for API section
DocMeta.setdocmeta!(OtsuThresholding, :DocTestSetup, :(using OtsuThresholding); recursive=true)

makedocs(;
    modules=[OtsuThresholding],
    authors="Eric Diaz <eric.diaz@gmail.com>",
    sitename="OtsuThresholding.jl",
    format=Documenter.HTML(;
        canonical="https://esd100.github.io/OtsuThresholding.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)



deploydocs(;
    repo="github.com/esd100/OtsuThresholding.jl",
    devbranch="main",
)
