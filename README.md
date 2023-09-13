# Aft-FFT
The Aft FFT library (C++).

## Rationale
The purpose of this library is to achieve DFT performance on-par with FFTW3, without archaic code generation. FFTW3
uses Caml-to-C code generation to implement performant FFT codelets. Aft attempts to replicate this performance using
only compile-time evaluation present in the C++ standard from C++11 (though Aft depends on C++17).

At the moment, only power-of-2 vector-length DFT is supported. Advanced planning for arbitrary-length vectors is not 
implemented. Aft uses decimation-in-time Cooley-Tukey.
