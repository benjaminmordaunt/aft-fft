/*
 * Copyright (c) 2023 Benjamin Mordaunt
 *
 * The Aft FFT library project.
 */

#include <iostream>
#include <gcem.hpp>
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
        explicit AftFFT(aft_complex<T> const (&in)[M]) : m_in(in), m_out{0} {}
        aft_complex<T> const (&m_in)[M];
        aft_complex<T> m_out[M];

        template<unsigned int N>
        __attribute__((always_inline)) inline void dft(unsigned int i, unsigned int stride) {
            T romega, iomega, er, ei, or_, oi, ur, ui;

            if (N == 1) {
                m_out[i].real = m_in[i].real;
                m_out[i].imag = m_in[i].imag;
                return;
            }

            dft<N/2>(i, stride*2);
            dft<N/2>(i+stride, stride*2);

            for (int j = 0; j < N/2; j++) {
                romega = gcem::cos(-j * M_2_PI / N);
                iomega = gcem::sin(-j * M_2_PI / N);
                er = m_out[i + 2*j * stride].real;
                ei = m_out[i + 2*j * stride].imag;
                or_ = m_out[i + (2*j + 1) * stride].real;
                oi = m_out[i + (2*j + 1) * stride].imag;
                ur = or_ * romega - oi * iomega;
                ui = or_ * iomega + oi * romega;
                m_out[i + 2*j * stride].real = er + ur;
                m_out[i + 2*j * stride].imag = ei + ui;
                m_out[i + (2*j + 1) * stride].real = er - ur;
                m_out[i + (2*j + 1) * stride].imag = ei - ui;
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