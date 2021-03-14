#ifndef MASKED_SPGEMM_INPUT_HELPERS_H
#define MASKED_SPGEMM_INPUT_HELPERS_H

template<class T>
auto createDistribution(T lo = 0, T hi = std::numeric_limits<T>::max()) {
    if constexpr (std::is_integral_v<T>) {
        return std::uniform_int_distribution<T>{lo, hi};
    } else if constexpr (std::is_floating_point_v<T>) {
        return std::uniform_real_distribution<T>{lo, hi};
    }
}

#endif //MASKED_SPGEMM_INPUT_HELPERS_H
