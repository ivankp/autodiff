#ifndef IVAN_DEBUG_HH
#define IVAN_DEBUG_HH

#ifndef NDEBUG
#include <iostream>

#define STR1(x) #x
#define STR(x) STR1(x)

#define TEST(var) std::cout << \
  "\033[33m" STR(__LINE__) ": " \
  "\033[36m" #var ":\033[0m " << (var) << std::endl;

#else

#define TEST(var)

#endif

#endif

template <typename> struct test_type;
#define TEST_TYPE(X) test_type<decltype(X)>{};
