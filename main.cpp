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
        explicit AftFFT(aft_complex<T> const (&in)[M], aft_complex<T> (&out)[M]) : m_in(in), m_out(out) {}
        aft_complex<T> const (&m_in)[M];
        aft_complex<T> (&m_out)[M];

        // dft<0U> will be dead-code-eliminated - but NOLINT marker required.
        template<unsigned int N>
        inline void dft_direct() {  // NOLINT(misc-no-recursion)
            T romega, iomega, er, ei, ur, ui;

#ifdef AFT_DEBUG
            // Current parameters for DFT.
            std::cout << "[" << N << "] Aft-DFT-Direct" << "\n";
#endif

            for_<N>([&] (auto k) {
                for_<N>([&](auto j) {
                    romega = force_consteval<T, to_bytes(gcem::cos<T>(-2 * j.value * k.value * M_PI / N))>;
                    iomega = force_consteval<T, to_bytes(gcem::sin<T>(-2 * j.value * k.value * M_PI / N))>;
                    er = m_in[j.value].real;
                    ei = m_in[j.value].imag;

#ifdef AFT_DEBUG
                    // Printing the twiddle factors
                    std::cout << "[" << N << "] romega: " << romega << " iomega: " << iomega << "\n";
#endif
                    ur = er * romega - ei * iomega;
                    ui = er * iomega + ei * romega;

                    m_out[k.value].real += ur;
                    m_out[k.value].imag += ui;
                });
            });
        }

        void run() {
            dft_direct<M>();
        }
    };
}

TEST(dft1d, l4) {
    using namespace aft;
    aft_complex<float> x[4] = {{5.f, 1.f},
                               {5.f, -1.f},
                               {5.f, 0.f},
                               {5.f, 0.f}};

    aft_complex<float> out[4] = {0};

    aft_complex<float> expected[4] = {{20.f, 0.f},
                                 {-1.f, 1.f},
                                 {0.f, 2.f},
                                 {1.f, 1.f},};

    auto planner = AftFFT(x, out); planner.run();
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(out[i].real, expected[i].real, 1e-5f);
        EXPECT_NEAR(out[i].imag, expected[i].imag, 1e-5f);
    }
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
