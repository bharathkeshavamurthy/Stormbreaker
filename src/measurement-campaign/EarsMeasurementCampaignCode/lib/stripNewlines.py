# Only remove \n and \r (if there is any).
#
# Yaguang Zhang, Purdue University, 2017-06-13

def stripStrArray(strArray):
    strArray = [x.strip('\n') for x in strArray]
    strArray = [x.strip('\r') for x in strArray]
    return strArray

def stripStr(s):
    s = s.strip('\n')
    s = s.strip('\r')
    return s

def strCmpStripped(s1, s2):
    return stripStr(s1) == stripStr(s2)