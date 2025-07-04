# LLRS

This is the official implementation of the LLRS cryptography scheme (``FS-LLRS: Lattice-Based Linkable Ring Signature With Forward Security for Cloud-Assisted Electronic Medical Records``) in Python programming language. 

This repository is a part of the [cryptography schemes](https://github.com/BatchClayderman/Cryptography-Schemes). 

### Option

- [/q|-q|q]: Specify that the following option is the value of $q$ (default: $256$). 
- [/n|-n|n]: Specify that the following option is the value of $n$ (default: $256$). 
- [/m|-m|m]: Specify that the following option is the value of $m$ (default: $4096$). 
- [/d|-d|d]: Specify that the following option is the value of $d$ (default: $10$). 
- [/k|-k|k]: Specify that the following option is the value of $k$ (default: $4$). 
- [/N|-N|N]: Specify that the following options are the values of $N$. 
- [/h|-h|h|/help|--help|help]: Show this help information. This option is not case-sensitive. 

### Exit Code
- ``EXIT_SUCCESS`` ($0$): The Python script finished successfully. 
- ``EXIT_FAILURE`` ($1$): The Python script finished not passing all the verifications. 
- ``EOF`` ($-1$): The Python script received unrecognized commandline options. 

### Note

1) All the commandline options are case-sensitive (except "[/h|-h|h|/help|--help|help]") and optional. 
2) The parameters $q$, $n$, $m$, $d$, and $k$ should be positive integers and will obey the following priority: values obtained from the command line > values specified by the user within the script > default values set within the script. 
3) The values of $N$ specified from the commandline will be directly appended to those specified by the user within the script. Each value of $N$ should be an integer that is larger than $1$. The unsatisfying ones will be set to $2$. 
4) The value of $q$ should be a 2-based integer that is larger than $2$. Otherwise, it will be defaulted to 256. 
5) The parameters $q$, $n$, $m$, and $k$ should meet the requirements that $m \leqslant 2n \log_2 q$, $4n | m$, and $2k | m$. If one or more of the requirements are not satisfied, they will be set to their default values respectively. 

### Citation

If you wish to cite this work, please use the following BibTeX. 

```
@article{chen2024fs,
  title={FS-LLRS: Lattice-based Linkable Ring Signature with Forward Security for Cloud-assisted Electronic Medical Records},
  author={Chen, Xue and Xu, Shiyuan and Gao, Shang and Guo, Yu and Yiu, Siu-Ming and Xiao, Bin},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2024},
  publisher={IEEE}
}
```

Thank you for your citations. 

## ``LLRS_v1.0.py``

In this version, the integer nullspace $***A******e*** = ***O*** (mod *q*)$ is solved by ``lstsq``, where the solution is a fuzzy one. 

### Usage

- ``python "LLRS_v1.0.py" [/q|-q|q] q [/n|-n|n] n [/m|-m|m] m [/d|-d|d] d [/k|-k|k] k [/N|-N|N] N1 N2 ...``
- ``python "LLRS_v1.0.py" [/h|-h|h|/help|--help|help]``

### Example

- ``python "LLRS_v1.0.py"``
- ``python "LLRS_v1.0.py" /q 256 /n 256 /m 4096``
- ``python "LLRS_v1.0.py" -q 256 -n 256 -m 4096 -d 10 -k 4``
- ``python "LLRS_v1.0.py" q 256 n 256 m 4096 d 10 k 4 N 2 4 8``
- ``python "LLRS_v1.0.py" --help``

## ``LLRS_v2.0.py``

In this version, the TrapGen child procedure is optimized based on the first version. 

### Usage

- ``python "LLRS_v2.0.py" [/q|-q|q] q [/n|-n|n] n [/m|-m|m] m [/d|-d|d] d [/k|-k|k] k [/N|-N|N] N1 N2 ...``
- ``python "LLRS_v2.0.py" [/h|-h|h|/help|--help|help]``

### Example

- ``python "LLRS_v2.0.py"``
- ``python "LLRS_v2.0.py" /q 256 /n 256 /m 4096``
- ``python "LLRS_v2.0.py" -q 256 -n 256 -m 4096 -d 10 -k 4``
- ``python "LLRS_v2.0.py" q 256 n 256 m 4096 d 10 k 4 N 2 4 8``
- ``python "LLRS_v2.0.py" --help``

## ``LLRS_v3.0.py``

This version is based on the first version, where the integer nullspace $***A******e*** = ***O*** (mod *q*)$ is solved according to [https://stackoverflow.com/questions/77932216/solve-integer-matrix-equation-ax-b-mod-q-for-x](https://stackoverflow.com/questions/77932216/solve-integer-matrix-equation-ax-b-mod-q-for-x). The solution should be exact, with more time required for computation. 

### Usage

- ``python "LLRS_v3.0.py" [/q|-q|q] q [/n|-n|n] n [/m|-m|m] m [/d|-d|d] d [/k|-k|k] k [/N|-N|N] N1 N2 ...``
- ``python "LLRS_v3.0.py" [/h|-h|h|/help|--help|help]``

### Example

- ``python "LLRS_v3.0.py"``
- ``python "LLRS_v3.0.py" /q 256 /n 256 /m 4096``
- ``python "LLRS_v3.0.py" -q 256 -n 256 -m 4096 -d 10 -k 4``
- ``python "LLRS_v3.0.py" q 256 n 256 m 4096 d 10 k 4 N 2 4 8``
- ``python "LLRS_v3.0.py" --help``

## ``LLRS_v4.0.py``

This version is based on the second version, where the integer nullspace $***A******e*** = ***O*** (mod *q*)$ is solved according to [https://stackoverflow.com/questions/77932216/solve-integer-matrix-equation-ax-b-mod-q-for-x](https://stackoverflow.com/questions/77932216/solve-integer-matrix-equation-ax-b-mod-q-for-x). The solution should be exact, with more time required for computation. 

### Usage

- ``python "LLRS_v4.0.py" [/q|-q|q] q [/n|-n|n] n [/m|-m|m] m [/d|-d|d] d [/k|-k|k] k [/N|-N|N] N1 N2 ...``
- ``python "LLRS_v4.0.py" [/h|-h|h|/help|--help|help]``

### Example

- ``python "LLRS_v4.0.py"``
- ``python "LLRS_v4.0.py" /q 256 /n 256 /m 4096``
- ``python "LLRS_v4.0.py" -q 256 -n 256 -m 4096 -d 10 -k 4``
- ``python "LLRS_v4.0.py" q 256 n 256 m 4096 d 10 k 4 N 2 4 8``
- ``python "LLRS_v4.0.py" --help``
