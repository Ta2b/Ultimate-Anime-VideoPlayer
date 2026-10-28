ajp_add_unit_tests(subs
  SOURCES
    atlas_test.cpp compat_test.cpp parity_test.cpp renderer_test.cpp srt_test.cpp support.cpp
    text_encoding_test.cpp thread_test.cpp track_test.cpp worker_test.cpp
  # parity_test compose la liste ASS_Image de libass directement.
  DEPS ajp::dep::ass)
