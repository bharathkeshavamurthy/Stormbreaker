if(GCC_NOT_5_2_0)
    if(CONFIG_NEWLIB_NANO_FORMAT)
        set(LIBC c_nano)
    else()
        set(LIBC c)
    endif()

    set(LIBM m)
else()
    if(CONFIG_SPIRAM_CACHE_WORKAROUND)
        set(LIBC c-psram-workaround)
        set(LIBM m-psram-workaround)
    else()
        if(CONFIG_NEWLIB_NANO_FORMAT)
            set(LIBC c_nano)
        else()
            set(LIBC c)
        endif()

        set(LIBM m)
    endif()
endif()