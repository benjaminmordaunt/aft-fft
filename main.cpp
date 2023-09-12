/*
 * Copyright (c) 2023 Benjamin Mordaunt
 *
 * The Aft FFT library project.
 */

#include <iostream>
#include <cmath>
#include <fftw3.h>

static constexpr bool ISP2(const unsigned int x) {
    return (x & (x - 1)) == 0;
}

namespace aft {
    template<typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
    struct aft_complex {
        T real;
        T imag;
    };

    template<unsigned int M, typename T>
    struct AftFFT {
        static_assert(ISP2(M), "Only power-of-2 DFT vector lengths supported.");
        explicit AftFFT(aft_complex<T> const (&in)[M]) : m_in(in) {}
        aft_complex<T> const (&m_in)[M];
        aft_complex<T> m_out[M];

        template<unsigned int N>
        __attribute__((always_inline)) inline void dft(unsigned int ii,
                                                       unsigned int oi) {
            T romega, iomega;

            if (N == 1) {
                m_out[oi] = m_in[ii];
                return;
            }

            dft<N/2>(ii, oi);
            dft<N/2>(ii+1, oi+1);

            for (int i = 0; i < N/2; i++) {
                romega = cos(-i * M_2_PI / N);
                iomega = sin(-i * M_2_PI / N);
                m_out[oi].real = m_in[ii].real + m_in[ii+1].real * romega;
                m_out[oi+N/2].real = m_in[ii].real - m_in[ii+1].real * romega;
                m_out[oi].imag = m_in[ii].imag + m_in[ii+1].imag * iomega;
                m_out[oi+N/2].imag = m_in[ii].imag - m_in[ii+1].imag * iomega;
                ii += 2;
                oi += 2;
            }
        }

        void run() {
            dft<M>(0, 1);
        }
    };
}

int main() {
    using namespace aft;

    const aft_complex<float> x[] = {{5.f, 0.f},
                                {5.f, 0.f},
                                {5.f, 0.f},
                                {5.f, 0.f}};
    auto plan = AftFFT(x);
    plan.run();

    for (auto &element : plan.m_out) {
        std::cout << element.real << std::endl;
    }
    return 0;
}