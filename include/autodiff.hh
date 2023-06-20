#ifndef IVAN_AUTODIFF_HH
#define IVAN_AUTODIFF_HH

#include <array>
#include <utility>
#include <type_traits>
#include <algorithm>
#include <cmath>

namespace ivan::autodiff {

template <typename T, size_t... I>
struct var {
  T x { };
  std::array<T,sizeof...(I)> d { };

  constexpr var() = default;
  constexpr var(const T& x) requires(sizeof...(I) == 0)
  : x(x) { }
  constexpr var(const T& x) requires(sizeof...(I) == 1)
  : x(x), d{1} { }

  explicit constexpr operator T() const { return x; }
};

template <typename T>
var(T) -> var<T>;

template <size_t I, typename T>
constexpr var<T,I> mkvar(T x) { return { x }; }

// ------------------------------------------------------------------

template <typename> struct var_sentinel;
template <typename T, size_t... I> struct var_sentinel<var<T,I...>> { };

template <size_t> struct index_constant { };

template <size_t J, size_t... I>
constexpr bool in = ( (I == J) || ... );

template <size_t J, size_t... I>
constexpr size_t index = in<J,I...> ? ( (I < J) + ... ) : size_t(-1);

// ------------------------------------------------------------------

template <size_t J, typename T, size_t... I>
T& d(var<T,I...>& v) requires (in<J,I...>) {
  return v.d[index<J,I...>];
}
template <size_t J, typename T, size_t... I>
const T& d(const var<T,I...>& v) requires (in<J,I...>) {
  return v.d[index<J,I...>];
}
template <size_t J, typename T, size_t... I>
T d(const var<T,I...>& v) requires (!in<J,I...>) {
  return { };
}

// ------------------------------------------------------------------

template <size_t... I>
constexpr auto index_set(auto apply) {
  constexpr auto arr_size = []{
    std::array<size_t,sizeof...(I)> arr { I... };
    std::sort( arr.begin(), arr.end() );
    return std::pair(
      arr,
      std::unique( arr.begin(), arr.end() ) - arr.begin()
    );
  }();
  return [&]<size_t... K>(std::index_sequence<K...>) {
    return apply( std::index_sequence< arr_size.first[K]... >{} );
  }(std::make_index_sequence<arr_size.second>{});
}

// ------------------------------------------------------------------

template <
  typename OP,
  typename A, size_t... I
>
constexpr auto unary_op( const var<A,I...>& a ) {
  using T = decltype( OP::f(a.x) );
  var<T,I...> v;
  v.x = OP::f(a.x);
  [&]<size_t... Ks>(var_sentinel<var<T,Ks...>>) {
    ([&]<size_t K>(index_constant<K>){
      d<K>(v) = OP::template dfda<K>(a);
    }(index_constant<Ks>{}), ... );
  }(var_sentinel<decltype(v)>{});
  return v;
}

#define IVAN_AUTODIFF_MAKE_UNARY_OP(NAME,IMPL) \
template <typename A, size_t... I> \
constexpr auto NAME( const var<A,I...>& a ) { \
  return unary_op<IMPL>(a); \
}

template <
  typename OP,
  typename A, typename B, size_t... I, size_t... J
>
constexpr auto binary_op( const var<A,I...>& a, const var<B,J...>& b ) {
  using T = decltype( OP::f(a.x,b.x) );
  using result_t = decltype(index_set<I...,J...>(
    []<size_t... K>(std::index_sequence<K...>) {
      return var<T,K...>{};
    }
  ));
  result_t v;
  v.x = OP::f(a.x,b.x);
  [&]<size_t... Ks>(var_sentinel<var<T,Ks...>>) {
    ([&]<size_t K>(index_constant<K>){
      if constexpr (in<K,I...>) d<K>(v) += OP::template dfda<K>(a,b);
      if constexpr (in<K,J...>) d<K>(v) += OP::template dfdb<K>(a,b);
    }(index_constant<Ks>{}), ... );
  }(var_sentinel<decltype(v)>{});
  return v;
}

#define IVAN_AUTODIFF_MAKE_BINARY_OP(NAME,IMPL) \
template <typename A, typename B, size_t... I, size_t... J> \
constexpr auto NAME( const var<A,I...>& a, const var<B,J...>& b ) { \
  return binary_op<IMPL>(a,b); \
} \
template <typename A, typename B, size_t... I> \
constexpr auto NAME( const var<A,I...>& a, const B& b ) { \
  return NAME(a,var(b)); \
} \
template <typename A, typename B, size_t... J> \
constexpr auto NAME( const A& a, const var<B,J...>& b ) { \
  return NAME(var(a),b); \
}

struct binary_plus_impl {
  static auto f(const auto& a, const auto& b) { return a + b; }
  template <size_t K>
  static auto dfda(const auto& a, const auto& b) { return d<K>(a); }
  template <size_t K>
  static auto dfdb(const auto& a, const auto& b) { return d<K>(b); }
};
IVAN_AUTODIFF_MAKE_BINARY_OP(operator+,binary_plus_impl)

struct binary_minus_impl {
  static auto f(const auto& a, const auto& b) { return a - b; }
  template <size_t K>
  static auto dfda(const auto& a, const auto& b) { return d<K>(a); }
  template <size_t K>
  static auto dfdb(const auto& a, const auto& b) { return -d<K>(b); }
};
IVAN_AUTODIFF_MAKE_BINARY_OP(operator-,binary_minus_impl)

struct binary_mult_impl {
  static auto f(const auto& a, const auto& b) { return a * b; }
  template <size_t K>
  static auto dfda(const auto& a, const auto& b) { return d<K>(a) * b.x; }
  template <size_t K>
  static auto dfdb(const auto& a, const auto& b) { return d<K>(b) * a.x; }
};
IVAN_AUTODIFF_MAKE_BINARY_OP(operator*,binary_mult_impl)

struct binary_div_impl {
  static auto f(const auto& a, const auto& b) { return a / b; }
  template <size_t K>
  static auto dfda(const auto& a, const auto& b) { return d<K>(a) / b.x; }
  template <size_t K>
  static auto dfdb(const auto& a, const auto& b) { return - d<K>(b) * (a.x / (b.x*b.x)); }
};
IVAN_AUTODIFF_MAKE_BINARY_OP(operator/,binary_div_impl)

struct unary_log_impl {
  static auto f(const auto& a) { using std::log; return log(a); }
  template <size_t K>
  static auto dfda(const auto& a) { return d<K>(a)/a.x; }
};
IVAN_AUTODIFF_MAKE_UNARY_OP(log,unary_log_impl)

// ------------------------------------------------------------------

template <typename T, size_t... I>
auto& operator << (auto& o, const var<T,I...>& v) { return o << v.x; }

}

#endif
