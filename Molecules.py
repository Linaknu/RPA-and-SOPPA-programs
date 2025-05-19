
from pyscf import gto


#---------------------------------------------------------------------------------------------------------
#Molecules from Geertsen et al. (1991)
#---------------------------------------------------------------------------------------------------------

#Ref. Graham et al. (1986) doi: 10.1063/1.451436
#No frozen core
Be = gto.Mole()
Be.build(
    atom = '''Be  0.0  0.0  0.0''',
    basis = {'Be': gto.basis.parse(
'''
Be    S
    3630.0000             0.000839
    532.300               0.006735
    117.80                0.035726
    32.660                0.138635
    10.480                0.385399
    3.668                 0.547688
Be    S
    3.668                 0.213406
    1.354                 0.814692
Be    S
    0.389                 1.0
Be    S
    0.1502                1.0
Be    S
    0.05241               1.0
Be    S
    0.0200                1.0
Be    S
    0.0080                1.0
Be    S
    0.003                 1.0
Be    S
    0.001                 1.0
Be    P
    6.710                 1.0
Be    P
    1.442                 1.0
Be    P
    0.4103                1.0
Be    P
    0.1397                1.0
Be    P
    0.04922               1.0
Be    P
    0.016                 1.0
Be    P
    0.007                 1.0
Be    P
    0.003                 1.0
Be    P
    0.001                 1.0
Be    D
    0.650                 1.0
Be    D
    0.200                 1.0
Be    D
    0.070                 1.0
Be    D
    0.015                 1.0
Be    D
    0.0032                1.0
''')},
        cart = True)

#CH+ ref. Larsson and Siegbahn (1983) doi: 10.1016/0301-0104(83)85030-7
#Freeze none
CH_plus = gto.Mole()
CH_plus.build(
    atom = '''C  0.0  0.0  0.0
              H  0.0  0.0  2.1324''',
    unit = 'AU',
    charge= 1,
    cart = True,
    basis = {'C': gto.basis.parse(
'''
C    S
    9471                  0.000776
    1398                  0.006218
    307.5                 0.033575
    84.54                 0.134278
    26.91                 0.393668
    9.409                 0.544169
C    S
    9.409                 0.248075
    3.500                 0.782844
C    S
    1.068                 1.0    
C    S
    0.60                  1.0
C    S
    0.25                  1.0
C    S
    0.10                  1.0
C    P
    25.37                 0.038802
    5.776                 0.243118
    1.787                 0.810162
C    P
    0.65                  1.0
C    P
    0.35                  1.0
C    P
    0.15                  1.0
C    P
    0.05                  1.0
C    D
    1.0                   1.0
C    D
    0.3                   1.0
'''),
'H': gto.basis.parse(
'''
H    S
    837.22                0.0001112
    123.524               0.000895
    27.7042               0.004737
    7.82599               0.019518
    2.56504               0.065862
    0.938258              0.178008
H    S
    0.372145              1.0
H    S
    0.155838              1.0
H    S
    0.066180              1.0
H    P
    2.12                  1.0
H    P
    0.77                  1.0
H    P
    0.28                  1.0
''')})


#---------------------------------------------------------------------------------------------------------
#Molecules in table II in Loos et al. (2021), DOI: 10.1063/5.0055994                         CC2,CCSD,CCSDTQP
#Geometries from Loos et al. (2018), DOI: 10.1021/acs.jctc.8b00406,     All geometries are optimized at the CC3/aug-cc-pVTZ level of theory
#---------------------------------------------------------------------------------------------------------

#Ref: Loos et al. (2018)
#Ref: CCSD, FCI: Loos et al. (2021) 
#Freeze 1
NH3_Loos = gto.MoleBase()
NH3_Loos.build(atom = '''N  0.12804615  -0.00000000  0.00000000
                    H  -0.59303935  0.88580079  -1.53425197
                    H  -0.59303935  -1.77160157  -0.00000000
                    H  -0.59303935  0.88580079  1.53425197 ''',
        unit = 'AU',
        symmetry =True,
        basis = 'aug-cc-pVDZ')

#Freeze 1
BH_Loos = gto.MoleBase()
BH_Loos .build(atom = '''B  0.00000000  0.00000000  0.00000000
                         H  0.00000000  0.00000000  2.31089693''',
        unit = 'AU',
        symmetry =True,
        basis = 'aug-cc-pVDZ')

#Freeze 2
BF_Loos = gto.MoleBase()
BF_Loos .build(atom = '''B  0.00000000  0.00000000  0.00000000
                         F  0.00000000  0.00000000  2.39729626''',
        unit = 'AU',
        symmetry =True,
        basis = 'aug-cc-pVDZ')


#Frozen = 2
CO_Loos = gto.MoleBase()
CO_Loos.build(atom = '''C  0.00  0.000  -1.24942055
                        O  0.00  0.000  0.89266692 ''',
        unit = 'AU',
        symmetry =True,
        basis = 'aug-cc-pVDZ')

#Freeze 2
N2_Loos = gto.MoleBase()
N2_Loos.build(
        atom = '''N 0.0 0.0 1.04008632; 
                  N 0.0 0.0 -1.04008632''',
        unit = 'AU',
        symmetry =True,
        basis = 'aug-cc-pVDZ')


#Freeze 5
HCl_Loos = gto.MoleBase()
HCl_Loos.build(
        atom = '''H 0.0 0.0 2.38483140; 
                  Cl 0.0 0.0 -0.02489783''',
        unit = 'AU',
        basis = 'aug-cc-pVDZ')


#Freeze 5
H2S_Loos = gto.MoleBase()
H2S_Loos.build(atom = '''S  0.00000000  0.00000000  -0.50365086
                    H  0.00000000  1.81828105  1.25212288
                    H  0.00000000  -1.81828105  1.25212288''',
        unit = 'AU',
        symmetry =True,
        basis = 'aug-cc-pVDZ')



#Freeze 1
H2O_Loos = gto.MoleBase()
H2O_Loos.build(atom = '''O 0.00000000 0.00000000 -0.13209669
                    H 0.00000000 1.43152878 0.97970006
                    H 0.00000000 -1.43152878 0.97970006''',
        unit = 'AU',
        symmetry =True,
        basis = 'aug-cc-pVDZ')



#---------------------------------------------------------------------------------------------------------
#Molecules in table 1 in Christiansen (1998)
#---------------------------------------------------------------------------------------------------------

#Ref. Christiansen et al. (1996) doi: 10.1016/0009-2614(96)00394-6
#Freeze 1
H2O_Christiansen = gto.MoleBase()
H2O_Christiansen.build(
        atom = '''O 0.0 0.0 0.0 
                  H 0.0  1.429937284 -1.107175113
                  H 0.0 -1.429937284 -1.107175113''',
        unit = 'AU',
        symmetry = True,
        basis = {'O':['cc-pVDZ', gto.basis.parse('''
O    S
     0.07896              1.0
O    P
     0.06856              1.0
''')],
        'H': ['cc-pVDZ', gto.basis.parse('''
H    S
     0.02974              1.0
''')]})



#Ref. Christiansen et al. (1996) doi: 10.1016/0009-2614(96)00394-6
#Freeze 2
N2_Christiansen = gto.MoleBase()
N2_Christiansen.build(
        atom = '''N 0.0 0.0 0.0; 
                  N 0 0.0 2.068''',
        unit = 'AU',
        basis = 'cc-pVDZ')


#Ref. Koch et al. (1995) doi: 10.1016/0009-2614(95)00914-p
#Freeze none
Ne_Christiansen = gto.MoleBase()
Ne_Christiansen.build(
        atom = '''Ne 0.0 0.0 0.0''',
        basis = {'Ne':['cc-pVDZ', gto.basis.parse('''
Ne    S
      0.04                1.0
Ne    P
      0.03                1.0
''')]})


#Ref. Koch et al. (1995) doi: 10.1016/0009-2614(95)00914-p
#Freeze none
CH2_Christiansen = gto.Mole()
CH2_Christiansen.build(
    atom = '''C    0.0        0.0      0.0
              H    1.644403   0.0      1.32213
              H   -1.644403   0.0      1.32213''',
    unit = 'AU',
    symmetry = True,
    basis = {'C': ['cc-pVDZ', gto.basis.parse('''
C    S
     0.015                1.0
''')],
'H': ['cc-pVDZ', gto.basis.parse('''
H    S
     0.025                1.0
''')]})

        

#ref. Larsen et al. (2001) Doi:
# see footnote on page 
#Freeze 0
BH_Christiansen = gto.MoleBase()
BH_Christiansen.build(atom = '''B  0.0   0.0    0.0
                          H  0.0   0.0    2.3289 ''',
        unit = 'AU',
        symmetry =True,
        basis = 'd-aug-cc-pVDZ')




#Ref. Larsen et al. (1999) doi: 10.1063/1.479460 
#Freeze 1
HF_Christiansen = gto.MoleBase()
HF_Christiansen.build(
        atom = '''H 0.0 0.0 0.0; 
                  F 0 0.0 1.7328795''',
        unit = 'AU',
        symmetry = True,
        basis = 'aug-cc-pVDZ')



#---------------------------------------------------------------------------------------------------------
#Test molecules. Not used in thesis
#---------------------------------------------------------------------------------------------------------

NH3Test = gto.MoleBase()
NH3Test.build(atom = '''N  0.12804615  -0.00000000  0.00000000
                    H  -0.59303935  0.88580079  -1.53425197
                    H  -0.59303935  -1.77160157  -0.00000000
                    H  -0.59303935  0.88580079  1.53425197 ''',
        unit = 'AU',
        symmetry =True,
        basis = 'sto-3g')

COTest = gto.MoleBase()
COTest.build(atom = '''C  0.00  0.000  -1.24942055
                   O  0.00  0.000  0.89266692 ''',
        unit = 'AU',
        symmetry =True,
        basis = 'sto-3g')

Benzene = gto.MoleBase()
Benzene.build(
        atom = ''' H      1.2194     -0.1652      2.1600
                   C      0.6825     -0.0924      1.2087
                   C     -0.7075     -0.0352      1.1973
                   H     -1.2644     -0.0630      2.1393
                   C     -1.3898      0.0572     -0.0114
                   H     -2.4836      0.1021     -0.0204
                   C     -0.6824      0.0925     -1.2088
                   H     -1.2194      0.1652     -2.1599
                   C      0.7075      0.0352     -1.1973
                   H      1.2641      0.0628     -2.1395
                   C      1.3899     -0.0572      0.0114
                   H      2.4836     -0.1022      0.0205''',
        unit = 'AU',
        basis = 'cc-pVDZ')