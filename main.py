import numpy as np
import io
from random import SystemRandom as Sysrand


class RandomWalk(object):
    """
    This class specifies defaults functions for the Random Walk algorithm
    plus the periodic boundary conditions
    """
    def __init__(self, numseed=1, numofiter=0,
                 maxmov=2, numofmols=5, boxsize=(30, 30, 30)):
        self.numofmols = numofmols
        self.numofiter = numofiter
        self.maxmov = maxmov
        self.posshape = (3, numofmols)
        self.boxsize = np.array(boxsize, dtype=int)
        self.initpos = self.randcoord()
        self.currentpos = self.initpos
        self.randseed = np.random.seed(numseed)

    def randcoord(self):
        """
        Functions generates initial random coordinates for the number of molecules/points.
        These coordinates are constrained by the |max{generated_coord}| <= min{spatial dimension of box},
        so the coordinates are generated  always inside the box.
        :return 3 x n nparray, where n is the number of molecules:
        """
        randvec = (np.random.randn(self.posshape[0], self.posshape[1])
                   - 0.5) * 2 * np.min(self.boxsize) / 2
        return randvec + self.pbccorr(randvec)

    def randmov(self):
        """Generates a random vector of size 3xn, where n is the number of molecules
        :return Random Vector; 3 x n nparraym, where n is the number of molecules:
        """
        pass

    def pbccorr(self, posit):
        """
        Corrects coordinates according to PBC box, it returns corrections
        instead of a corrected coordinates.
        :param Coordinates; 3xn nparraym, where n is the number of molecules:
        :return PBC correction matrix; 3 x n nparray, where n is the number of molecules:
        """
        pbc = self.boxsize.reshape(3, 1) / 2
        pbcboolpos = (posit - pbc) > 0
        pbcboolneg = (posit + pbc) < 0
        pbccorr = pbcboolneg * pbc - pbcboolpos * pbc
        return pbccorr

    def randwalkstep(self, randmovvect):
        """
        Performs one randomwalk iteration with the PBC corrected coordinates,
        it requires a random vector as its argument.
        :param Random vector; 3 x n nparray, where n is the number of molecules:
        :return None:
        """
        newcoord = self.currentpos + randmovvect * self.maxmov
        self.currentpos = newcoord + self.pbccorr(newcoord)

    def randwalk3d(self):
        """
        Performs iterations of the Random Walk algorithm
        :return None:
        """
        for step in range(self.numofiter):
            randmovvect = self.randmov()
            self.randwalkstep(randmovvect)


class MolecularRandWalk(RandomWalk):
    def __init__(self, numseed=0, numofiter=0, maxmov=2,
                 numofmols=5, boxsize=(30, 30, 30), bondformprob=0.5,
                 bondbreakprob=0.1, bondlenform=2, multibond=False):
        super().__init__(numseed, numofiter,
                         maxmov, numofmols, boxsize)
        self.bondformprob = bondformprob
        self.bondbreakprob = bondbreakprob
        self.randobj = Sysrand(numseed)
        self.bondlenform = bondlenform
        self.multibond = multibond
        self.currentbondmat = {x: np.array([], dtype=int) for x in range(1, self.numofmols + 1)}
        self.emptyarray = np.array([])
        self.atomindices = np.arange(self.posshape[1]).reshape(1, self.posshape[1]) + 1

    def randmov(self):
        """
        Generates a random vector of size 3xn, where n is the number of molecules,
        the method checks if the bond can be formed or broken based on the probabilities
        and bonding distance provided before the start of the simulation.
        :return Random Vector; 3 x n nparraym, where n is the number of molecules:
        """
        randvec = (np.random.rand(self.posshape[0], self.posshape[1]) - 0.5) * 2
        ifform = self.ifbondeddict()
        currentbondmattemp = self.currentbondmat
        # Breaking bonds: id does preserve bonds with probabilit 1 - bodnbreakprob
        # and it does not remove the pair from the dictionary, so the bond is not formed again
        for key in range(1, self.numofmols + 1):
            assert(key != 0)
            ifbondbreak = self.currentbondmat[key]
            if 1 - self.bondbreakprob >= self.randobj.random() and ifbondbreak.size > 0:
                value = ifbondbreak[0]
                randvec[0, key - 1] = randvec[0, value - 1]
                randvec[1, key - 1] = randvec[1, value - 1]
                randvec[2, key - 1] = randvec[2, value - 1]
            elif ifbondbreak.size > 0:
                value = ifbondbreak[0]
                self.currentbondmat[key] = self.emptyarray
                self.currentbondmat[value] = self.emptyarray

        # forming new bonds, without previously generated involved
        for key in range(1, self.numofmols + 1):
            assert(key != 0)
            if currentbondmattemp[key].size > 0:
                continue
            ifbondform = np.setdiff1d(ifform[key], currentbondmattemp[key])
            for value in ifbondform:
                assert(value != 0)
                # There is a little problem: it changes the random value twice
                # but it does not change overall results, it could be a tip
                # for performance improvments
                if self.bondformprob >= self.randobj.random() and currentbondmattemp[value].size == 0:
                    randvec[0, key - 1] = randvec[0, value - 1]
                    randvec[1, key - 1] = randvec[1, value - 1]
                    randvec[2, key - 1] = randvec[2, value - 1]
                    self.currentbondmat[key] = np.array([value], dtype=int)
                    self.currentbondmat[value] = np.array([key], dtype=int)
                    keyatomcoord = self.currentpos[:, key - 1]
                    assatomcoord = self.currentpos[:, value - 1]
                    distass = np.linalg.norm(keyatomcoord - assatomcoord)
                    rattio = self.bondlenform / distass
                    self.currentpos[:, value - 1] = keyatomcoord * (1 - rattio) + assatomcoord * rattio
                    break
        return randvec

    def randmovagr(self):
        """
        Generates a random vector of size 3xn, where n is the number of molecules,
        the method checks if the bond can be formed or broken based on the probabilities
        and bonding distance provided before the start of the simulation.
        :return Random Vector; 3 x n nparraym, where n is the number of molecules:
        """
        randvec = (np.random.rand(self.posshape[0], self.posshape[1]) - 0.5) * 2
        ifform = self.ifbondeddict()
        # Make sure that the forming/breaking bonds events are indepedentent
        currentbondmattemp = self.currentbondmat
        # Breaking bonds: id does preserve bonds with probabilit 1 - bodnbreakprob
        # and it does not remove the pair from the dictionary, so the bond is not formed again
        for key in range(1, self.numofmols + 1):
            assert(key != 0)
            ifbondbreak = self.currentbondmat[key]
            for value in ifbondbreak:
                if 1 - self.bondbreakprob >= self.randobj.random() and ifbondbreak.size > 0:
                    randvec[0, value - 1] = randvec[0, key - 1]
                    randvec[1, value - 1] = randvec[1, key - 1]
                    randvec[2, value - 1] = randvec[2, key - 1]
                elif ifbondbreak.size > 0:
                    self.currentbondmat[key] = np.setdiff1d(self.currentbondmat[key], np.array([value]))
                    self.currentbondmat[value] = np.setdiff1d(self.currentbondmat[value], np.array([key]))

        # forming new bonds, without previously generated involved
        for key in range(1, self.numofmols + 1):
            assert(key != 0)
            ifbondform = np.setdiff1d(ifform[key], currentbondmattemp[key])
            for value in ifbondform:
                assert(value != 0)
                # There is a little problem: it moves every atom once, so every dimer is moved twice
                if self.bondformprob >= self.randobj.random() and currentbondmattemp[value].size == 0:
                    randvec[0, value - 1] = randvec[0, key - 1]
                    randvec[1, value - 1] = randvec[1, key - 1]
                    randvec[2, value - 1] = randvec[2, key - 1]
                    self.currentbondmat[key] = np.append(self.currentbondmat[key], value)
                    self.currentbondmat[value] = np.append(self.currentbondmat[value], key)
                    # Normalize the bond lenght
                    keyatomcoord = self.currentpos[:, key - 1]
                    assatomcoord = self.currentpos[:, value - 1]
                    distass = np.linalg.norm(keyatomcoord - assatomcoord)
                    rattio = self.bondlenform / distass
                    self.currentpos[:, value - 1] = keyatomcoord * (1 - rattio) + assatomcoord * rattio
        return randvec

    @staticmethod
    def distmat(coordmat):
        """
        Calculates distance matrix for the provided matrix.
        :param coordinate nparray of m x n shape:
        :return distance nparray n x n:
        """
        coordmat = coordmat.T
        return np.sqrt(np.sum((coordmat - coordmat[:, np.newaxis])**2, axis=2))

    def distmattobool(self, distmatrix):
        """
        Changes all entries of the distance matrix to the bool values according
        to the used the bond formation length set on the beginning of the simulation.
        :param n x n nparray:
        :return n x n nparray filled with bool values:
        """
        return distmatrix <= self.bondlenform

    def matbooltopairlist(self, matbool):
        """
        Checks based on the distance matrix filled with bool values
        if two points/molecules can form bond based on the bond length
        provided on the beginning of the simulation. It retunrs
        dicitonary, which define atoms allowed to be bonded.
        :param n x n nparray filled with the bool values:
        :return dictionary with two keys, atomnumber_1: atomnumber_1, atomnumber_2 ...:
        """
        molnum = self.numofmols
        indicemat = (matbool * self.atomindices).flatten()
        ifbonded = {}
        for molind in range(molnum):
            fromind = molind * (molnum + 1) + 1
            toind = molind * molnum + molnum
            newboolmat = indicemat[fromind:toind]
            ifbonded[molind + 1] = newboolmat[newboolmat != 0]
        return ifbonded

    def ifbondeddict(self):
        """
        Wraps three methods, distmat, distmattobool and
        matbooltopairlist.
        :return: dictionary with two keys, atomnumber: atomnumber:
        """
        distmat = self.distmat(self.currentpos)
        matbool = self.distmattobool(distmat)
        return self.matbooltopairlist(matbool)


class RunProgram(MolecularRandWalk, ):
    def __init__(self, numseed=0,
                 numofiter=0, maxmov=2, numofmols=5, boxsize=(30, 30, 30),
                 bondformprob=0.5, bondbreakprob=0.1, bondlenform=2, multibond=False,
                 outputname='default.xyz', savingfreq=1, buffersize=1000):
        super().__init__(numseed, numofiter, maxmov, numofmols, boxsize,
                         bondformprob, bondbreakprob, bondlenform, multibond)
        self.outputname = outputname
        self.savingfreq = savingfreq
        self.savingcounter = savingfreq
        self.atomnames = self.getatomnames()
        self.outputobject = open(self.outputname, 'w', -1)
        self.bytestream = io.BytesIO()
        self.outbuffer = []
        self.buffersize = buffersize

    def getatomnames(self):
        """
        Returns list of atom names used in the simulation,
        it is only required for the visualisation purpose.
        :return list of characters of length n, where n is number of molecules:
        """
        lenofarray = self.posshape[1]
        return ['C'] * lenofarray

    def mattoxyz(self, text):
        """
        Creates string containing coordinates to xyz format.
        :param String with coordinates:
        :return xyz format string:
        """
        xyzinp = '' + str(self.numofmols) + '\n' + 'Comment' + '\n'
        for i in range(len(text)):
            xyzinp += self.atomnames[i] + ' ' + text[i] + '\n'
        return xyzinp

    def savetofile(self):
        """
        Saves file from the buffer list by using mattoxyz
        method and writing to file.
        :return None:
        """
        outbuffer = self.outbuffer
        xyzstring = ''
        for currpos in outbuffer:
            rawtext = [' '.join('%0.3f' %x for x in y) for y in currpos.T]
            xyzstring += self.mattoxyz(rawtext)
        self.outputobject.write(xyzstring)
        self.outbuffer = []

    def savetobuffer(self):
        """
        Saves coordinates nparray to buffer list.
        :return None:
        """
        self.outbuffer.append(self.currentpos)

    def randwalk3d(self):
        """
        Perform iterations of the Random Walk algorithm, further
        the overriding method method implements saving to the
        buffer and file with certain frequency.
        :return None:
        """
        savingcounter = self.savingfreq
        savetofile = self.buffersize
        for step in range(self.numofiter):
            if self.savingcounter == self.savingfreq:
                print("### Iteration number of %d ###" % step)
                self.savetobuffer()
                if self.buffersize == savetofile:
                    self.savetofile()
                    savetofile = 0
                savingcounter = 0
                savetofile += 1
            randmovvect = self.randmovagr()
            self.randwalkstep(randmovvect)
            savingcounter += 1

    def runsim(self):
        """
        Wraps the randwalk3d method makes sure to flush the buffer
        and close the file object.
        :return None:
        """
        self.randwalk3d()
        self.savetofile()
        self.outputobject.close()


if __name__ == '__main__':
    runsim = RunProgram(numofiter=10456, numofmols=12, bondformprob=0.999,
                        bondbreakprob=0.001, maxmov=2, boxsize=[30, 30, 30])
    runsim.runsim()




