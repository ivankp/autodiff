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

  template <size_t J>
  static constexpr bool has = ( (I == J) || ... );

  template <size_t J>
  static constexpr size_t index = has<J> ? ( (I < J) + ... ) : size_t(-1);

  template <size_t J> requires ( has<J> )
  auto& get() noexcept { return d[index<J>]; }
  template <size_t J> requires ( has<J> )
  const auto& get() const noexcept { return d[index<J>]; }

  void compute_derivatives(auto&& f) {
    [&]<size_t... Ks>(std::index_sequence<Ks...>) {
      ( f(std::index_sequence<Ks>{}), ... );
    }(std::index_sequence<I...>{});
  };
};

template <typename T>
var(T) -> var<T>;

template <size_t I, typename T>
constexpr var<T,I> mkvar(T x) { return { x }; }

// ------------------------------------------------------------------

template <typename A, typename B, size_t... I, size_t... J>
constexpr auto merge_var_types( var<A,I...>, var<B,J...> ) {
  using T = std::common_type_t<A,B>;
  constexpr auto arr_size = []{
    std::array<size_t,sizeof...(I)+sizeof...(J)> arr { I..., J... };
    std::sort( arr.begin(), arr.end() );
    return std::pair(
      arr,
      std::unique( arr.begin(), arr.end() ) - arr.begin()
    );
  }();
  return [&]<size_t... K>(std::index_sequence<K...>) {
    return var< T, arr_size.first[K]... >{};
  }(std::make_index_sequence<arr_size.second>{});
}

template <typename A, typename B>
constexpr auto merge_var_types( A a, B b ) {
  return merge_var_types( var(a), var(b) );
}

template <typename A, typename B>
using merge_var_t = decltype(merge_var_types(
  std::declval<A>(),
  std::declval<B>()
));

// ------------------------------------------------------------------

template <size_t J, typename T>
constexpr size_t index = size_t(-1);

template <size_t J, typename T, size_t... I>
constexpr size_t index<J,var<T,I...>> =
  ( (I == J) || ... ) ? ( (I < J) + ... ) : size_t(-1);

// ------------------------------------------------------------------
template <typename A, typename B, size_t... I>
constexpr auto operator+( const var<A,I...>& a, const B& b ) {
  return a + var(b);
}
template <typename A, typename B, size_t... J>
constexpr auto operator+( const A& a, const var<B,J...>& b ) {
  return var(a) + b;
}
template <typename A, typename B, size_t... I, size_t... J>
constexpr auto operator+( const var<A,I...>& a, const var<B,J...>& b ) {
  using result_t = decltype(merge_var_types(a,b));
  result_t v;
  v.x = a.x + b.x;
  v.compute_derivatives(
    [&]<size_t K>(std::index_sequence<K>){
      if constexpr (a.template has<K>) v.template get<K>() += a.template get<K>();
      if constexpr (b.template has<K>) v.template get<K>() += b.template get<K>();
    }
  );
  return v;
}

template <typename A, typename B, size_t... I>
constexpr auto operator*( const var<A,I...>& a, const B& b ) {
  return a * var(b);
}
template <typename A, typename B, size_t... J>
constexpr auto operator*( const A& a, const var<B,J...>& b ) {
  return var(a) * b;
}
template <typename A, typename B, size_t... I, size_t... J>
constexpr auto operator*( const var<A,I...>& a, const var<B,J...>& b ) {
  using result_t = decltype(merge_var_types(a,b));
  result_t v;
  v.x = a.x * b.x;
  v.compute_derivatives(
    [&]<size_t K>(std::index_sequence<K>){
      if constexpr (a.template has<K>) v.template get<K>() += a.template get<K>() * b.x;
      if constexpr (b.template has<K>) v.template get<K>() += b.template get<K>() * a.x;
    }
  );
  return v;
}

template <typename A, size_t... I>
constexpr auto log( const var<A,I...>& a ) {
  var<A,I...> v;
  using std::log;
  v.x = log(a.x);
  v.compute_derivatives(
    [&]<size_t K>(std::index_sequence<K>){
      v.template get<K>() += a.template get<K>() / a.x;
    }
  );
  return v;
}

// ------------------------------------------------------------------

template <typename T, size_t... I>
auto& operator << (auto& o, const var<T,I...>& v) { return o << v.x; }

}

#endif
