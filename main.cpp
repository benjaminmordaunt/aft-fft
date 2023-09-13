/*
 * Copyright (c) 2023 Benjamin Mordaunt
 *
 * The Aft FFT library project.
 */

#include <cmath>
#include <iostream>
#include <type_traits>
#include <gcem.hpp>
#include <gtest/gtest.h>

namespace aft {
    static constexpr bool IsPowerOf2(const unsigned int x) {
        return (x & (x - 1)) == 0;
    }

    template <typename T>
    using byte_representation = std::array<std::byte, sizeof(T)>;

    template <typename T>
    constexpr byte_representation<T> to_bytes(const T& value) {
        return std::bit_cast<byte_representation<T>>(value);
    }

    template <typename T>
    constexpr T from_bytes(const byte_representation<T>& bytes) {
        return std::bit_cast<T>(bytes);
    }

    template <typename T, byte_representation<T> Bytes>
    static constexpr auto force_consteval = from_bytes<T>(Bytes);

    template<std::size_t N>
    struct num { static const constexpr int value = N; };

    template <class F, std::size_t... Is>
    void for_(F func, std::index_sequence<Is...>)
    {
        using expander = int[];
        (void)expander{0, ((void)func(num<Is>{}), 0)...};
    }

    template <std::size_t N, typename F>
    void for_(F func)
    {
        for_(func, std::make_index_sequence<N>());
    }

    template<typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
    struct aft_complex {
        T real;
        T imag;
    };

    template<unsigned int M, typename T>
    struct AftFFT {
        static_assert(IsPowerOf2(M), "Only power-of-2 DFT vector lengths supported.");
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

            for_<N/2>([&] (auto j) {
                romega = force_consteval<T, to_bytes(gcem::cos<T>(-j.value * M_2_PI / N))>;
                iomega = force_consteval<T, to_bytes(gcem::sin<T>(-j.value * M_2_PI / N))>;
                er = m_out[i + 2*j.value * stride].real;
                ei = m_out[i + 2*j.value * stride].imag;
                or_ = m_out[i + (2*j.value + 1) * stride].real;
                oi = m_out[i + (2*j.value + 1) * stride].imag;
                ur = or_ * romega - oi * iomega;
                ui = or_ * iomega + oi * romega;
                m_out[i + 2*j.value * stride].real = er + ur;
                m_out[i + 2*j.value * stride].imag = ei + ui;
                m_out[i + (2*j.value + 1) * stride].real = er - ur;
                m_out[i + (2*j.value + 1) * stride].imag = ei - ui;
            });
        }

        void run() {
            dft<M>(0, 1);
        }
    };
}

TEST(dft1d, l4) {
    using namespace aft;
    aft_complex<float> x[8] = {{5.f, 0.f},
                               {5.f, 0.f},
                               {5.f, 0.f},
                               {5.f, 0.f},
                               {5.f, 0.f},
                               {5.f, 0.f},
                               {5.f, 0.f},
                               {5.f, 0.f}};

    aft_complex<float> exp[8] = {{40.f, 0.f},
                                 {0.f, 0.f},
                                 {0.f, 0.f},
                                 {0.f, 0.f},
                                 {0.f, 0.f},
                                 {0.f, 0.f},
                                 {0.f, 0.f},
                                 {0.f, 0.f}};

    auto planner = AftFFT(x); planner.run();
    for (int i = 0; i < 8; i++) {
        EXPECT_EQ(planner.m_out[i].real, exp[i].real);
        EXPECT_EQ(planner.m_out[i].imag, exp[i].imag);
    }
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
