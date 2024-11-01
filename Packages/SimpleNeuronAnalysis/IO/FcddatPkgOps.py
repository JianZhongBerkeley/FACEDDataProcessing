import os
import numpy as np
import re


def listPkgNames(srcDir):
    assert(os.path.exists(srcDir))
    pkgNameSuffix = "_Pkg"
    subDirNameList = [iSubDirName for iSubDirName in os.listdir(srcDir) if os.path.isdir(srcDir + "\\" + iSubDirName)]
    pkgNameList = [iSubDirName for iSubDirName in subDirNameList if iSubDirName[-len(pkgNameSuffix):] == pkgNameSuffix]
    return pkgNameList


def sortPkgNames(src_pkgNameList):
    PkgNumStrPrefix = r"_F"
    pkgNumStrSuffix = "_Pkg"
    pkgNumRe = PkgNumStrPrefix + r"[0-9]+" + pkgNumStrSuffix
    src_pkgNames = np.array(src_pkgNameList)
    pkgNums = np.zeros(src_pkgNames.shape)
    for iPkg in range(len(src_pkgNames)):
        pkgName = src_pkgNames[iPkg]
        pkgNum = int(re.findall(pkgNumRe, pkgName)[0][len(PkgNumStrPrefix):-len(pkgNumStrSuffix)])
        pkgNums[iPkg] = pkgNum
    return src_pkgNames[np.argsort(pkgNums)]



def listFACEDTifDataPaths(pkgPath):
    dataDirName = "Data"
    dataDirPath = pkgPath + "\\" + dataDirName
    assert(os.path.exists(dataDirPath))
    srcTifFileNameList = [fileName for fileName in os.listdir(dataDirPath) if os.path.splitext(fileName)[-1] == ".tif" ]
    srcTifFileNameList.sort()
    srcTifFilePathList = [dataDirPath + "\\" + fileName for fileName in srcTifFileNameList]
    return srcTifFilePathList


def getPkgFNum(pkg_name):
    F_num_regex = r"F\d+_Pkg"
    F_num_str = re.findall(F_num_regex, pkg_name)[0]
    F_num = int(F_num_str[1:-4])
    return F_num
