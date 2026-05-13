PySCF implementation of rung-3.5 density functionals and interpretive tools. 

Density functionals are implemented as r35dft, an extension of the dft module 

Density functionals include the Minnesota functional M11po. The semilocal piece
of this functional is implemented in a fork of libxc available on the gitlab
repository bjanesko/libxc-r35dft . Use XC functionals xc='HYB_MGGA_X_M11PO,MGGA_C_M11PO' 

Int
=========================

2026-05-08

* Version 0.1

Install
-------
* Recommended install is in a python virtual environment created in /your/top/dir/
```
python3 -m venv venv
```
To enter the virtual environment, type:

```
source /your/top/dir/venv/bin/activate
```

Recommended install in virtual environment
```
pip install pyscf
pip install git+https://github.com/bjanesko/pyscf-r35dft
```

To use M11po, download and compile the libxc-r35dft fork of libxc, using the
standard autoreconf procedure, then copy the library to your new pyscf environment 
```
./configure --prefix=/your/libxc/install/dir --enable-shared
make
make install
cp /your/libxc/install/dir/lib/libxc.so /your/top/dir/venv/lib/python3.10/site-packages/pyscf/lib/deps/lib/
```

You can find more details of extended modules in the document
[extension modules](http://pyscf.org/pyscf/install.html#extension-modules)
