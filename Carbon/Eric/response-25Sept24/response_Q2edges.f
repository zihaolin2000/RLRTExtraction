      PROGRAM response_Q2edges

      IMPLICIT NONE

      real*8 Z, A, Q2, W2, xb, qv, nu, dnu, F1, FL, RT, RL, RTE, RLE
      real*8 nuel, ex, RTQE, RLQE, RTIE, RLIE, RTNS, RLNS, RTTOT, RLTOT
      real*8 flNS, f1NS, fLt, f1t, mp/0.938273/
      integer i,j,type
      integer io_status, arg_status, unit
      character(len=30) filename
      real*8 xvalc(100) /     
     & 0.14878E+00,0.35188E+01,0.19212E+00,0.10157E+02,0.78215E+00,
     & 0.12534E+01,0.29829E+01,0.15909E+01,0.47777E+00,-.31209E+01,
     & 0.78402E+00,0.37948E+00,0.80540E+00,0.58939E+01,0.26046E+02,
     & -.77121E+01,0.12921E+00,0.20000E+00,0.20215E+00,0.44084E+00,
     & 0.21875E+00,0.63072E-01,0.24000E+00,0.80769E-14,0.26131E+00,
     & 0.18972E-03,0.50393E+02,-.45088E-01,0.34620E-01,0.19080E+00,
     & 0.21216E+01,0.12687E+00,0.13677E+00,0.10000E+01,0.12570E+00,
     & 0.50987E-01,0.49457E+00,-.47081E+00,0.46722E+02,0.20000E+01,
     & 0.10000E+01,0.95515E+00,0.10726E+01,0.98427E+00,0.92995E+00,
     & 0.10078E+01,0.95592E+00,0.10190E+01,0.97183E+00,0.97687E+00,
     & 0.10462E+01,0.99248E+00,0.10000E+01,0.10350E+01,0.10106E+01,
     & 0.94178E+00,0.10872E+01,0.11774E+01,0.11709E+01,0.10000E+01,
     & 0.10087E+01,0.10478E+01,0.97778E+00,0.97386E+00,0.98669E+00,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01 /

      
      A = 12.0
      Z = 6.0

!       read(5,*) q2
      call get_command_argument(1, filename, arg_status)
      unit = 20
      open(UNIT=unit, FILE=filename, STATUS='old', IOSTAT=io_status)
      
      if (io_status/= 0) then
        print *, 'Unable to open file:',filename
        stop
      endif

      
      ! dnu = 0.00015
      ! nu = 0.0
      i = 0

      do 
        read(unit,*,IOSTAT=io_status) i, q2, nu
        if (io_status /= 0) exit
      ! do i=1,3000
        ! nu = nu + dnu

c        q2 = qv*qv-nu*nu
        nuel = q2/2./(0.931494*A)
        w2 = mp*mp+2.0*mp*nu-q2
        xb = q2/2.0/mp/nu

        ex = nu-nuel
        
        type = 1
        call csfitcomp(w2,q2,A,Z,XVALC,type,f1,fL) !!!  total response
        fL = 2.0*xb*fL
        RTTOT = 2.0/mp*F1/1000.0
        RLTOT = qv*qv/q2/2.0/mp/xb*FL/1000.

        type = 2
        call csfitcomp(w2,q2,A,Z,XVALC,type,f1,fL) !!!  QE response
        fL = 2.0*xb*fL
        RTQE = 2.0/mp*F1/1000.0
        RLQE = qv*qv/q2/2.0/mp/xb*FL/1000.0
        
        type = 3
        call csfitcomp(w2,q2,A,Z,XVALC,type,f1,fL) !!!  IE response
        fL = 2.0*xb*fL
        RTIE = 2.0/mp*F1/1000.0
        RLIE =  qv*qv/q2/2.0/mp/xb*FL/1000.0
        
        type = 4
        call csfitcomp(w2,q2,A,Z,XVALC,type,f1,fL) !!!  TE response
        fL = 2.0*xb*fL  
        RTE = 2.0/mp*F1/1000.0
        RLE = 0.0


c        write(6,*) RLTOT,RLIE+RLQE
        
        fLNS = 0.0
        f1NS = 0.0
        do j=2,22
           call nuc12sf(Z,A,nu,q2,j,f1t,fLt)

          fLNS = fLNS + fLt
          f1NS = f1NS + f1t      
        enddo
        RTNS = 2.0/mp*F1NS/1000.0 
        RLNS =  qv*qv/q2/2.0/mp/xb*FLNS/1000.0

        ! if(ex.LE.0.012) then  !!! Only needed for plotting purposes
        !    RTNS = RTNS/6.0
        !    RLNS = RLNS/6.0
        ! endif
        if(RLNS.LE.1E-40) RLNS = 0.0
        if(RTNS.LE.1E-40) RTNS = 0.0
        
        RLTOT = RLTOT+RLNS
        RTTOT = RTTOT+RTNS
        
        if(q2.GT.0.0) 
     &       write(6,2000) qv,q2,ex,nu,RTTOT,RLTOT,RTQE,RLQE,RTIE,RLIE,
     &                      RTE,RLE,RTNS,RLNS          
 
        
      enddo

!  2000  format(4f9.5,10E11.3)
 2000  format(4f9.5,10E15.7)
      

      return
      end


      
      
      
      
     
CCC-----------------

      
