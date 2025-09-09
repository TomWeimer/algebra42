# ./.julia_config/startup.jl
using Pkg

# Activate the current workspace (local folder)
Pkg.activate(".")

# Add the package(s) you need
Pkg.add("Infiltrator")


using Infiltrator


include("../test/allTests.jl")

exit() 

