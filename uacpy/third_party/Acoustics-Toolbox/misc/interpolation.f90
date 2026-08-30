MODULE interpolation

CONTAINS
  SUBROUTINE interp1( x, y, xi, yi )

    ! Given a set of points (x, y), computes the interpolated value at xi
    ! using piecewise linear interpolation

    ! Assumes x is monotonically increasing

    IMPLICIT NONE
    REAL (KIND=8), INTENT( IN  ) :: x( : ), y( : ), xi( : )
    REAL (KIND=8), INTENT( OUT ) :: yi( : )
    INTEGER                      :: N, Ni, I, iseg
    REAL (KIND=8)                :: R

    N    = SIZE( x  )
    Ni   = SIZE( xi )
    iseg = 1

    ! loop over the interpolation points
    DO I = 1, Ni
       ! search for the bracketing pair of tabulated values. The segment
       ! index runs 1 : N-1, and each loop must stop once iseg is clamped
       ! or a query outside [ x(1), x(N) ] spins forever.
       DO WHILE ( xi( I ) > x( iseg + 1 ) .AND. iseg < N - 1 )  ! to the right of the current segment?
          iseg = iseg + 1
       END DO

       DO WHILE ( xi( I ) < x( iseg     ) .AND. iseg > 1     )  ! to the left  of the current segment?
          iseg = iseg - 1
       END DO

       ! proportional distance between points, clamped so a query outside
       ! [ x(1), x(N) ] holds the end value instead of extrapolating off
       ! the table (extrapolating a level table can go negative)
       R       = ( xi( I ) - x( iseg ) ) / ( x( iseg + 1 ) - x( iseg ) )
       R       = MIN( MAX( R, 0.0D0 ), 1.0D0 )
       yi( I ) = ( 1.0 - R ) * y( iseg ) + R * y( iseg + 1 )
    END DO

  END SUBROUTINE interp1
END MODULE interpolation
