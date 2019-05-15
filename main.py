from randmol import RunProgram

if __name__ == '__main__':
    simulation = RunProgram(numofiter=5000, numofmols=20, outputname='default.xyz')
simulation.runsim()