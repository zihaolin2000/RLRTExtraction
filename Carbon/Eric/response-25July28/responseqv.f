      PROGRAM RESPONSEQV

      IMPLICIT NONE

      real*8 Z, A, Q2, W2, xb, qv, nu, dnu, F1, FL, RT, RL, RTE, RLE
      real*8 nuel, ex, RTQE, RLQE, RTIE, RLIE, RTNS, RLNS, RTTOT, RLTOT
      real*8 flNS, f1NS, fLt, f1t, mp/0.938273/
      integer i,j,type
      real*8 xvalc(100) /     
     & 0.29477E+00,0.24674E+01,0.29248E+00,0.93155E+01,0.88366E+00,
     & 0.12912E+01,0.29829E+01,0.14371E+01,0.61395E+00,-.30962E+01,
     & 0.81209E+00,0.28001E+00,0.83370E+00,0.55909E+01,0.13605E+02,
     & -.79936E+01,0.12921E+00,0.20000E+00,0.20215E+00,0.44084E+00,
     & 0.21875E+00,0.17747E+00,0.22500E+00,0.11155E-01,0.22531E+00,
     & 0.10000E-05,0.00000E+00,-.40076E-01,0.27140E-01,0.11625E+00,
     & 0.15175E+01,0.10179E+00,0.10647E+00,0.10000E+01,0.36152E-01,
     & 0.42619E-01,0.00000E+00,0.00000E+00,0.29141E+02,0.20000E+01,
     & 0.96366E+00,0.96476E+00,0.10760E+01,0.99291E+00,0.93522E+00,
     & 0.10136E+01,0.96433E+00,0.10267E+01,0.97636E+00,0.99071E+00,
     & 0.99513E+00,0.10000E+01,0.10157E+01,0.10350E+01,0.10216E+01,
     & 0.93795E+00,0.98752E+00,0.10270E+01,0.11007E+01,0.10000E+01,
     & 0.10107E+01,0.99640E+00,0.99726E+00,0.10377E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01/

      
      A = 12.0
      Z = 6.0

      read(5,*) qv
      
      
      dnu = 0.00015
      nu = 0.0
      
      do i=1,3000
        nu = nu + dnu

        q2 = qv*qv-nu*nu
c        qv = sqrt(q2+nu*nu)
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
        do j=2,21
           call nuc12sf(Z,A,nu,q2,j,f1t,fLt)

          fLNS = fLNS + fLt
          f1NS = f1NS + f1t      
        enddo
        RTNS = 2.0/mp*F1NS/1000.0 
        RLNS =  qv*qv/q2/2.0/mp/xb*FLNS/1000.0

c        if(ex.LE.0.012) then  !!! Only needed for plotting purposes
c           RTNS = RTNS/6.0
c           RLNS = RLNS/6.0
c        endif
        
        RLTOT = RLTOT+RLNS
        RTTOT = RTTOT+RTNS
        
        if(q2.GT.0.0.AND.ex.GE.-0.01) 
     &       write(6,2000) qv,q2,ex,nu,RTTOT,RLTOT,RTQE,RLQE,RTIE,RLIE,
     &                      RTE,RLE,RTNS,RLNS          
 
        
      enddo

 2000  format(4f9.5,10E11.3)
      

      return
      end


      
      
      
      
     
CCC-----------------

      
