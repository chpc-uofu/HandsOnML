-- -*- lua -*-

local version = "25.9.1"

help([[ miniforge 
        Miniforge provides installers for the commands conda and mamba.
]])

whatis("Name         : miniforge ")
whatis("Version      : " .. version)
whatis("Category     : python, conda, mamba, pip")
whatis("Description  : ")
whatis("URL          : https://github.com/conda-forge/miniforge")
whatis("Installed on : 11/23/2025")
whatis("Modified on  : --- ")
whatis("Installed by : WRC ")

local home = os.getenv("HOME")
local base = pathJoin(home,"software/pkg", myModuleName(), version)

prepend_path("PATH", pathJoin(base,"bin"))
prepend_path("MANPATH", pathJoin(base,"share/man"))
family("python")
