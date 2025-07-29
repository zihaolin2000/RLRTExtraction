      PROGRAM RESPONSEQ2

      IMPLICIT NONE

      real*8 Z, A, Q2, W2, xb, qv, nu, dnu, F1, FL, RT, RL, RTE, RLE
      real*8 nuel, ex, RTQE, RLQE, RTIE, RLIE, RTNS, RLNS, RTTOT, RLTOT
      real*8 flNS, f1NS, fLt, f1t, mp/0.938273/
      integer i,j,type
      real*8 xvalc(100) /     
     & 0.26772E+00,0.26059E+01,0.27687E+00,0.93714E+01,0.88053E+00,
     & 0.12917E+01,0.29829E+01,0.14342E+01,0.61061E+00,-.30853E+01,
     & 0.87375E+00,0.27156E+00,0.81003E+00,0.37002E+01,0.12485E+02,
     & -.50000E+01,0.12921E+00,0.20000E+00,0.20215E+00,0.44084E+00,
     & 0.21875E+00,0.17698E+00,0.22500E+00,0.10712E-01,0.22531E+00,
     & 0.10000E-05,0.00000E+00,-.36258E-01,0.28133E-01,0.13302E+00,
     & 0.20134E+01,0.10288E+00,0.10694E+00,0.10000E+01,0.36092E-01,
     & 0.45207E-01,0.00000E+00,0.00000E+00,0.26861E+02,0.20000E+01,
     & 0.96504E+00,0.96501E+00,0.10760E+01,0.99283E+00,0.93425E+00,
     & 0.10131E+01,0.96279E+00,0.10264E+01,0.97717E+00,0.99118E+00,
     & 0.99633E+00,0.10000E+01,0.10153E+01,0.10350E+01,0.10225E+01,
     & 0.93949E+00,0.98713E+00,0.10262E+01,0.11018E+01,0.10000E+01,
     & 0.10111E+01,0.10002E+01,0.10038E+01,0.10472E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01/

      
      A = 12.0
      Z = 6.0

      read(5,*) q2
      
      
      dnu = 0.00015
      nu = 0.0
      
      do i=1,3000
        nu = nu + dnu

c        q2 = qv*qv-nu*nu
        qv = sqrt(q2+nu*nu)
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

        if(ex.LE.0.012) then  !!! Only needed for plotting purposes
           RTNS = RTNS/6.0
           RLNS = RLNS/6.0
        endif
        
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

      
