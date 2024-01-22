from sys import argv, exit
from copy import deepcopy
from datetime import datetime
from math import log2
from pytz import timezone
from time import sleep, time
try:
	from numpy import arange, array, concatenate, dot, eye, fill_diagonal, kron, sum as np_sum, tile, where, zeros # dot, kron, and np_sum require " % q"
	from numpy.linalg import lstsq # lstsq requires " % q"
	from numpy.random import randint
except Exception as e:
	print("Please install the library named \"numpy\" properly before this script can be run. ")
	print("Exception(s): ")
	print(e)
	print("Please press enter key to exit. ")
	input()
	exit(-1)
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EOF = (-1)
SCRIPT_NAME = "LLRS_v2.0.py"
DEFAULT_Q = 256
DEFAULT_N = 256
DEFAULT_M = 4096
DEFAULT_D = 10
DEFAULT_K = 4
parameters_dict = {} # {"q":16, "n":16, "m":256}
Ns = [2, 4, 6, 8, 16, 32] # Users should only modify the parameters here in this line and the previous line to accomplish more experiments. 


# Class #
class PARS:
	def __init__(self, N = 2, q = DEFAULT_Q, n = DEFAULT_N, m = DEFAULT_M, d = DEFAULT_D, k = DEFAULT_K, isAvailable = False, **extra_pars):
		if isinstance(N, int) and N > 1:
			self.__N = N
		else:
			print("The input N is not a positive integer that is larger than 1. It is defaulted to 2. ")
			self.__N = 2
		if isinstance(q, int) and q > 2 and log2(q) == int(log2(q)):
			self.__q = q
		else:
			print("The input q is not a 2-based positive integer that is larger than 2. It is defaulted to {0}. ".format(DEFAULT_Q))
			self.__q = DEFAULT_Q
		if isinstance(n, int) and n > 0:
			self.__n = n
		else:
			print("The input n is not a positive integer. It is defaulted to {0}. ".format(DEFAULT_N))
			self.__n = DEFAULT_N
		if isinstance(m, int) and m > 0:
			self.__m = m
		else:
			print("The input m is not a positive integer. It is defaulted to {0}. ".format(DEFAULT_M))
			self.__m = DEFAULT_M
		if isinstance(d, int) and d > 0:
			self.__d = d
		else:
			print("The input d is not a positive integer. It is defaulted to {0}. ".format(DEFAULT_D))
			self.__d = DEFAULT_D
		if isinstance(k, int) and k > 0:
			self.__k = k
		else:
			print("The input k is not a positive integer. It is defaulted to {0}. ".format(DEFAULT_K))
			self.__k = DEFAULT_K
		if self.__m < (self.__n << 1) * log2(self.__q) or self.__m % (self.__n << 2) != 0 or self.__m % (self.__k << 1) != 0:
			print("The input q, n, m, and k should meet the requirements that \"m >= 2n lb q\", \"4n | m\", and \"2k | m\". Nonetheless, one or more of the requirements are not satisfied. They are defaulted to {0}, {1}, {2} and {3} respectively. ".format(DEFAULT_Q, DEFAULT_N, DEFAULT_M, DEFAULT_K))
			self.__q = DEFAULT_Q
			self.__n = DEFAULT_N
			self.__m = DEFAULT_M
			self.__k = DEFAULT_K
		self.isAvailable = isAvailable if isinstance(isAvailable, bool) else False
		self.__sk = zeros((0, self.__m, self.__m), dtype = "int")
		self.__pk = zeros((0, self.__n, self.__m), dtype = "int")
		if extra_pars:
			print("Extra parameters for setting up are detected, listed as follows. \n{0}\n\n*** Please check the global parameter dictionary. ***\n".format(list(extra_pars.keys())))
	def getUpperN(self) -> int:
		return self.__N
	def getQ(self) -> int:
		return self.__q
	def getLowerN(self) -> int:
		return self.__n
	def getM(self) -> int:
		return self.__m
	def getD(self) -> int:
		return self.__d
	def getK(self) -> int:
		return self.__k
	def setC(self, C:array) -> None:
		self.__C = C
	def getC(self) -> array:
		return self.__C
	def setT(self, T:array) -> None:
		self.__T = T
	def getT(self) -> array:
		return self.__T
	def setB(self, B:array) -> None:
		self.__B = B
	def getB(self) -> array:
		return self.__B
	def setA_0i(self, A_0i:array) -> None:
		self.__A_0i = A_0i
	def getA_0i(self) -> array:
		return self.__A_0i
	def setT_A0i(self, T_A0i:array) -> None:
		self.__T_A0i = T_A0i
	def getT_A0i(self) -> array:
		return self.__T_A0i
	def setSk(self, sk:array) -> None:
		self.__sk = sk
	def getSk(self) -> array:
		return self.__sk
	def setPk(self, pk:array) -> None:
		self.__pk = pk
	def getPk(self) -> array:
		return self.__pk
	def setMiu(self, miu:array) -> None:
		self.__miu = miu
	def getMiu(self) -> array:
		return self.__miu
	def setL(self, l:int) -> None:
		self.__l = l
	def getL(self) -> int:
		return self.__l
	def setSigma(self, sigma:tuple) -> None:
		self.__sigma = sigma
	def getSigma(self) -> tuple:
		return self.__sigma
	def printVars(self, vars:list) -> None:
		if type(vars) not in (tuple, list):
			vars = [str(vars)]
		undefined = []
		for var in vars:
			var_name = "_PARS__{0}".format(var)
			if hasattr(self, var_name):
				print("{0} = {1}".format(var, getattr(self, var_name)))
			else:
				undefined.append(var)
		if undefined:
			print("Undefined variables: {0}".format(undefined))


# Child Functions #
def TrapGen(pars:PARS) -> tuple:
	q = pars.getQ()
	n = pars.getLowerN()
	m = pars.getM()
	g = 1 << arange(0, m // (n << 2)) # size = (m / 4n)
	G = kron(eye(n, dtype = "int"), g) % q # size = (n, m / 4)
	B = randint(q, size = (n, m >> 2)) # size = (n, m / 4)
	R = randint(2, size = (m >> 2, m >> 2)) # size = (m / 4, m / 4)
	A_0i = concatenate((B, (dot(B, R) % q + G) % q), axis = 1) # size = (n, m / 2)
	T_g = zeros((m // (n << 2), m // (n << 2)), dtype = "int") # size = (m / 4n, m / 4n)
	fill_diagonal(T_g, 2)
	fill_diagonal(T_g[1:], -1)
	T_G = kron(eye(n, dtype = "int"), T_g) % q # size = (m / 4, m / 4)
	G_ = G.T # size = (m / 4, n)
	T_Aa = concatenate(((eye(m >> 2, dtype = "int") + dot(dot(R, G_) % q, B) % q) % q, dot(-R, T_G) % q), axis = 1) # size = (m / 4, m / 2)
	T_Ab = concatenate((dot((-G_) % q, B) % q, T_G), axis = 1) # size = (m / 4, m / 2)
	T_A0i = concatenate((T_Aa, T_Ab), axis = 0) # size = (m / 2, m / 2)
	return (A_0i, T_A0i) # (size = (n, m / 2), size = (m / 2, m / 2))

def ExtBasis(F_B0:array, T_B0:array, B_0:array, q:int) -> array:
	W = lstsq(B_0, F_B0, rcond = None)[0].astype("int") % q
	T = concatenate((concatenate((T_B0, W), axis = 1), concatenate((zeros((W.shape[1], T_B0.shape[1]), dtype = "int"), eye(W.shape[1], dtype = "int")), axis = 1)), axis = 0)
	return T

def GenSample(pars:PARS, C_miu) -> array:
	N = pars.getUpperN()
	q = pars.getQ()
	n = pars.getLowerN()
	m = pars.getM()
	pk = pars.getPk() # size = (N, n, m)
	l = pars.getL()
	e = randint(q, size = (N + 1, m)) # size = (N + 1, m)
	z = zeros((n, 1), dtype = "int") # size = (n, 1)
	for i in list(range(l)) + list(range(l + 1, N)):
		z -= dot(pk[i, :, :], e[i, None].T) % q
		z %= q
	z -= dot(C_miu, e[N, None].T) % q
	z %= q
	return e


# Procedure Functions #
def Setup(N:int, pars_dict:dict) -> PARS:
	_pars_dict = pars_dict.copy()
	_pars_dict["N"] = N
	pars = PARS(**_pars_dict)
	q = pars.getQ()
	n = pars.getLowerN()
	m = pars.getM()
	d = pars.getD()
	k = pars.getK()
	C = randint(q, size = (d + 1, n, m)) # size = (d + 1, n, m) 
	pars.setC(C)
	T = randint(2, size = (k << 1, m >> 1, m // (k << 1))) # size = (2k, m / 2, m / 2k)
	pars.setT(T)
	B = randint(q, size = (6, n, m)) # size = (6, n, m)
	pars.setB(B)
	try:
		A_0i, T_A0i = TrapGen(pars) # (size = (n, m / 2), size = (m / 2, m / 2))
		pars.isAvailable = True
	except Exception as e:
		print("Exception(s) occurred in TrapGen of Setup, generating A_0i and T_A0i randomly. ")
		print("Exception(s): ")
		print(e)
		A_0i, T_A0i = randint(q, size = (n, m >> 1)), randint(q, size = (m >> 1, m >> 1)) # (size = (n, m / 2), size = (m / 2, m / 2))
		pars.isAvailable = False
	pars.setA_0i(A_0i)
	pars.setT_A0i(T_A0i)
	pars.printVars(["N", "q", "n", "m", "d", "k", "C", "T", "B"])
	return pars

def KeyExtract(pars:PARS) -> tuple:
	q = pars.getQ()
	n = pars.getLowerN()
	m = pars.getM()
	k = pars.getK()
	T = pars.getT() # size = (2k, m / 2, m / 2k)
	A_0i = pars.getA_0i() # size = (n, m / 2)
	T_A0i = pars.getT_A0i() # size = (m / 2, m / 2)
	tau_i = randint(2, size = (k, )) # size = (k, 1)
	start_time = time()
	F_tau_i = T[tau_i[0]] # size = (m / 2, m / 2)
	for i in range(1, k):
		F_tau_i = concatenate((F_tau_i, T[(i << 1) + tau_i[i]]), axis = 1)
	g_k = 1 << arange(0, m // (n << 1)) # size = (1, m / 2n)
	G_k = kron(eye(n, dtype = "int"), g_k) % q # size = (n, m / 2)
	B_0i = concatenate((A_0i, (dot(A_0i, F_tau_i) % q + G_k) % q), axis = 1) # size = (n, m)
	pk_i = B_0i # size = (n, m)
	T_gk = zeros((m // (n << 1), m // (n << 1)), dtype = "int") # size = (m / 2n, m / 2n)
	fill_diagonal(T_gk, 2)
	fill_diagonal(T_gk[1:], -1)
	T_Gk = kron(eye(n, dtype = "int"), T_gk) % q # size = (m / 2, m / 2)
	G_k_ = G_k.T # size = (m / 2, n)
	T_Ba = concatenate((eye(m >> 1, dtype = "int") + dot(dot(F_tau_i, G_k_) % q, A_0i) % q, dot(F_tau_i, T_Gk) % q), axis = 1) # size = (m / 2, m)
	T_Bb = concatenate((dot((-G_k_) % q, A_0i) % q, T_Gk), axis = 1) # size = (m / 2, m)
	T_B0i = concatenate((T_Ba, T_Bb), axis = 0) # size = (m, m)
	sk_i = T_B0i # size = (m, m)
	sk = concatenate((pars.getSk(), sk_i[None, :, :]), axis = 0) # size = (N, m, m)
	pars.setSk(sk)
	pk = concatenate((pars.getPk(), pk_i[None, :, :]), axis = 0) # size = (N, n, m)
	pars.setPk(pk)
	end_time = time()
	return (pars, end_time - start_time)

def KeyUpdate(pars:PARS) -> tuple:
	N = pars.getUpperN()
	q = pars.getQ()
	B = pars.getB()
	sk = pars.getSk() # size = (N, m, m)
	pk = pars.getPk() # size = (N, n, m)
	l = randint(N)
	pars.setL(l)
	B_0 = pk[l, :, :] # size = (n, m)
	F_001 = concatenate((B_0, B[0, :, :], B[2, :, :], B[5, :, :]), axis = 1) # size = (n, 4m)
	F_01 = concatenate((B_0, B[0, :, :], B[3, :, :]), axis = 1) # size = (n, 3m)
	F_1 = concatenate((B_0, B[1, :, :]), axis = 1) # size = (n, 2m)
	F_011 = concatenate((B_0, B[0, :, :], B[3, :, :], B[5, :, :]), axis = 1) # size = (n, 4m)
	F_101 = concatenate((B_0, B[1, :, :], B[2, :, :], B[5, :, :]), axis = 1) # size = (n, 4m)
	F_11 = concatenate((B_0, B[1, :, :], B[3, :, :]), axis = 1) # size = (n, 3m)
	F_111 = concatenate((B_0, B[1, :, :], B[3, :, :], B[5, :, :]), axis = 1) # size = (n, 4m)
	T_B0 = sk[l, :, :] # size = (m, m)
	T_001 = ExtBasis(F_001, T_B0, B_0, q)
	T_01 = ExtBasis(F_01, T_B0, B_0, q)
	T_1 = ExtBasis(F_1, T_B0, B_0, q)
	T_011 = ExtBasis(F_011, T_B0, B_0, q)
	T_101 = ExtBasis(F_101, T_B0, B_0, q)
	T_11 = ExtBasis(F_11, T_B0, B_0, q)
	T_111 = ExtBasis(F_111, T_B0, B_0, q)
	start_time = time()
	sk = (				\
		(T_001, T_01, T_1), 		\
		(T_01, T_1), 		\
		(T_011, T_1), 		\
		(T_1, ), 			\
		(T_101, T_11), 		\
		(T_11, ), 			\
		(T_111, )			\
	)
	t = randint(7)
	print("sk_t = {0}".format(sk[t]))
	end_time = time()
	return (pars, end_time - start_time)
	
def Sign(pars:PARS) -> tuple:
	N = pars.getUpperN()
	q = pars.getQ()
	n = pars.getLowerN()
	d = pars.getD()
	C = pars.getC() # size = (d + 1, n, m) 
	sk = pars.getSk() # size = (N, m, m)
	pk = pars.getPk() # size = (N, n, m)
	miu = concatenate((zeros((1, ), dtype = "int"), randint(2, size = (d, )))) # size = (1 + d)
	pars.setMiu(miu)
	C_miu = np_sum(where(miu == 0, 1, -1)[:, None, None] * C, axis = 0) % q # size = (n, m)
	pk_R = pk[0, :, :]
	start_time = time()
	for i in range(1, N):
		pk_R = concatenate((pk_R, pk[i]), axis = 1)
	pk_R = concatenate((pk_R, C_miu), axis = 1) # size = (n, (N + 1)m)
	l = pars.getL()
	e = GenSample(pars, C_miu) # size = (N + 1, m)
	H = np_sum(pk, axis = 0) % q # size = (n, m)
	tag = dot(H, sk[l, :, :]) % q # size = (n, m)
	sigma = (e, tag) # (size = (N + 1, m), size = (n, m))
	pars.setSigma(sigma)
	pars.printVars(["sigma"])
	end_time = time()
	return (pars, end_time - start_time)

def Verify(pars:PARS) -> tuple:
	N = pars.getUpperN()
	q = pars.getQ()
	n = pars.getLowerN()
	m = pars.getM()
	C = pars.getC() # size = (d + 1, n, m) 
	pk = pars.getPk() # size = (N, n, m)
	miu = pars.getMiu() # size = (1 + d)
	sigma = pars.getSigma() # (size = (N + 1, m), size = (n, m))
	start_time = time()
	e = sigma[0] # size = (N + 1, m)
	C_miu = np_sum(where(miu == 0, 1, -1)[:, None, None] * C, axis = 0) % q # size = (n, m)
	if 0 <= len(e) <= 8 * ((n + 1) * m) ** 0.5 and (np_sum(pk * e[:, None, None], axis = 0) % q == zeros((N, n, m), dtype = "int")).all():
		print("accept")
		end_time = time()
		return (True, end_time - start_time)
	else:
		print("reject" if pars.isAvailable else "unavailable")
		end_time = time()
		return (False, end_time - start_time)

def Link(pars1:PARS, pars2:PARS) -> bool:
	miu1 = pars1.getMiu() # size = (1 + d)
	pk1 = pars1.getPk() # size = (N, n, m)
	sigma1 = pars1.getSigma()
	tag1 = sigma1[-1] # size = (n, m)
	miu2 = pars2.getMiu() # size = (1 + d)
	pk2 = pars2.getPk() # size = (N, n, m)
	sigma2 = pars2.getSigma()
	tag2 = sigma2[-1] # size = (n, m)
	if (tag1 == tag2).all():
		print("link")
		return True
	else:
		print("unlink")
		return False


# Main Functions #
def getCurrentTime() -> str:
	tz = timezone("Asia/Shanghai")
	current_time = datetime.now(tz)
	return "{0} {1}".format(current_time.strftime("%Y/%m/%d %H:%M:%S"), current_time.tzname())

def printHelp() -> None:
	print("\"{0}\": A Python script for implementing LLRS, which is optimized by improving the TrapGen child procedure. ".format(SCRIPT_NAME), end = "\n\n")
	print("Option: ")
	print("\t[/q|-q|q]: Specify that the following option is the value of q (default: {0}). ".format(DEFAULT_Q))
	print("\t[/n|-n|n]: Specify that the following option is the value of n (default: {0}). ".format(DEFAULT_N))
	print("\t[/m|-m|m]: Specify that the following option is the value of m (default: {0}). ".format(DEFAULT_M))
	print("\t[/d|-d|d]: Specify that the following option is the value of d (default: {0}). ".format(DEFAULT_D))
	print("\t[/k|-k|k]: Specify that the following option is the value of k (default: {0}). ".format(DEFAULT_K))
	print("\t[/N|-N|N]: Specify that the following options are the values of N. ", end = "\n\n")
	print("Format: ")
	print("\tpython \"{0}\" [/q|-q|q] q [/n|-n|n] n [/m|-m|m] m [/d|-d|d] d [/k|-k|k] k [/N|-N|N] N1 N2 ...".format(SCRIPT_NAME))
	print("\tpython \"{0}\" [/h|-h|h|/help|--help|help]".format(SCRIPT_NAME), end = "\n\n")
	print("Example: ")
	print("\tpython \"{0}\"".format(SCRIPT_NAME))
	print("\tpython \"{0}\" /q {1} /n {2} /m {3}".format(SCRIPT_NAME, DEFAULT_Q, DEFAULT_N, DEFAULT_M))
	print("\tpython \"{0}\" -q {1} -n {2} -m {3} -d {4} -k {5}".format(SCRIPT_NAME, DEFAULT_Q, DEFAULT_N, DEFAULT_M, DEFAULT_D, DEFAULT_K))
	print("\tpython \"{0}\" [/q|-q|q] q [/n|-n|n] n [/m|-m|m] m [/d|-d|d] d [/k|-k|k] k [/N|-N|N] N1 N2 ...".format(SCRIPT_NAME))
	print("\tpython \"{0}\" [/h|-h|h|/help|--help|help]".format(SCRIPT_NAME), end = "\n\n")
	print("Exit code: ")
	print("\t{0}\tThe Python script finished successfully. ".format(EXIT_SUCCESS))
	print("\t{0}\tThe Python script finished not passing all the verifications. ".format(EXIT_FAILURE))
	print("\t{0}\tThe Python script received unrecognized commandline options. ".format(EOF), end = "\n\n")
	print("Note: ")
	print("\t1) All the commandline options are case-sensitive (except \"[/h|-h|h|/help|--help|help]\") and optional. ")
	print("\t2) The parameters q, n, m, d, and k should be positive integers and will obey the following priority: values obtained from the commandline > values specified by the user within the script > default values set within the script. ")
	print("\t3) The values of N specified from the commandline will be directly appended to those specified by the user within the script. Each value of N should be an integer that is larger than 1. The unsatisfying ones will be set to 2. ")
	print("\t4) The value of q should be a 2-based integer that is larger than 2. Otherwise, it will be defaulted to {0}. ".format(DEFAULT_Q))
	print("\t5) The parameters q, n, m, and k should meet the requirements that \"m >= 2n lb q\", \"4n | m\", and \"2k | m\". If one or more of the requirements are not satisfied, they will be set to their default values respectively. ", end = "\n\n")

def handleCommandline() -> dict:
	for arg in argv[1:]:
		if arg.lower() in ("/h", "-h", "h", "/help", "--help", "help", "/?", "-?", "?"):
			printHelp()
			return True
	commandline_dict = {}
	pointer = None
	for arg in argv[1:]:
		if arg in ("/q", "-q", "q"):
			pointer = "q"
		elif arg in ("/n", "-n", "n"):
			pointer = "n"
		elif arg in ("/m", "-m", "m"):
			pointer = "m"
		elif arg in ("/d", "-d", "d"):
			pointer = "d"
		elif arg in ("/k", "-k", "k"):
			pointer = "k"
		elif arg in ("/N", "-N", "N"):
			pointer = "N"
		elif pointer is None:
			print("Error handling commandline, please check your commandline or use \"/help\" for help. ")
			return False
		elif "N" == pointer:
			commandline_dict.setdefault("N", [])
			commandline_dict["N"].append(arg)
		else:
			commandline_dict[pointer] = arg
			pointer = None # reset
	for key in ("q", "n", "m", "d", "k"):
		try:
			if key in commandline_dict:
				commandline_dict[key] = int(commandline_dict[key])
		except:
			print("Error regarding {0} as an integer. Please check your commandline. ".format(key))
			return False
	try:
		if "N" in commandline_dict:
			commandline_dict["N"] = [int(item) for item in commandline_dict["N"]]
	except:
		print("Error regarding all the values in N as integers. Please check your commandline. ".format(key))
		return False
	return commandline_dict

def preExit(countdownTime = 5) -> None:
	try:
		cntTime = int(countdownTime)
		length = len(str(cntTime))
	except:
		return
	print()
	while cntTime > 0:
		print("\rProgram ended, exiting in {{0:>{0}}} second(s). ".format(length).format(cntTime), end = "")
		try:
			sleep(1)
		except:
			print("\rProgram ended, exiting in {{0:>{0}}} second(s). ".format(length).format(0))
			return
		cntTime -= 1
	print("\rProgram ended, exiting in {{0:>{0}}} second(s). ".format(length).format(cntTime))

def main() -> int:
	if not argv[0].endswith(SCRIPT_NAME):
		print("Warning: This Python script should be named \"{0}\". However, it is currently specified as another name. ".format(SCRIPT_NAME))
	commandlineArgument = handleCommandline()
	if isinstance(commandlineArgument, bool):
		return EXIT_SUCCESS if commandlineArgument else EOF
	print("Program named \"{0}\" started at {1}. ".format(SCRIPT_NAME, getCurrentTime()))
	if commandlineArgument:
		print("Parameters resolved from commandline: {0}".format(commandlineArgument))
		if "N" in commandlineArgument:
			Ns.extend(commandlineArgument["N"])
			del commandlineArgument["N"]
		parameters_dict.update(commandlineArgument)
	print("Parameters: {0}".format(parameters_dict))
	print("Ns: {0}".format(Ns), end = "\n" * 3)
	bRet = True
	dicts = {}
	skipped_N = []
	for N in Ns:
		if isinstance(N, int) and N >= 2:
			print("/** N = {0} **/".format(N))
			
			print("/* Setup */")
			start_time = time()
			pars = Setup(N, parameters_dict)
			end_time = time()
			dicts.setdefault(N, {"Setup":end_time - start_time})
			print()
			
			print("/* KeyExtract */")
			totalTimeCost = 0
			for i in range(N):
				pars, timeCost = KeyExtract(pars)
				totalTimeCost += timeCost
			pars.printVars(["sk", "pk"])
			dicts[N]["KeyExtract"] = totalTimeCost
			print()
			
			print("/* KeyUpdate */")
			pars, dicts[N]["KeyUpdate"] = KeyUpdate(pars)
			pars1 = deepcopy(pars)
			pars2 = deepcopy(pars)
			print()
			
			print("/* Sign1 */")
			pars1, dicts[N]["Sign1"] = Sign(pars1)
			print()
			
			print("/* Verify1 */")
			status, dicts[N]["Verify1"] = Verify(pars1)
			bRet = bRet and status
			print()
			
			print("/* Sign2 */")
			pars2, dicts[N]["Sign2"] = Sign(pars2)
			print()
			
			print("/* Verify2 */")
			status, dicts[N]["Verify2"] = Verify(pars2)
			bRet = bRet and status
			print()
			
			print("/* Link */")
			start_time = time()
			Link(pars1, pars2)
			end_time = time()
			dicts[N]["Link"] = end_time - start_time
			print()
		else:
			skipped_N.append(N)
	if skipped_N:
		if 1 == len(skipped_N):
			print("The following value of \"N\" has been skipped as it is not an integer that is larger than 2. \n{0}\n".format(skipped_N))
		else:
			print("The following values of \"N\" have been skipped as they are not integers that are larger than 2. \n{0}\n".format(skipped_N))
	if 0 == len(dicts):
		print("No experimental results are collected. ")
	elif 1 == len(dicts):
		print("The experimental result of the time consumption in seconds is shown as follows. ")
		print(dicts)
	else:
		print("The experimental results of the time consumption in seconds are shown as follows. ")
		print(dicts)
	preExit()
	print("\n\n\nProgram ended at {0} with exit code {1}. ".format(getCurrentTime(), EXIT_SUCCESS if bRet else EXIT_FAILURE))
	return EXIT_SUCCESS if bRet else EXIT_FAILURE



if __name__ == "__main__":
	exit(main())