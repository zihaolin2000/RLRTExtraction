      PROGRAM RESPONSEQ2

      IMPLICIT NONE

      real*8 Z, A, Q2, W2, xb, qv, nu, dnu, F1, FL, RT, RL, RTE, RLE
      real*8 nuel, ex, RTQE, RLQE, RTIE, RLIE, RTNS, RLNS, RTTOT, RLTOT
      real*8 flNS, f1NS, fLt, f1t, mp/0.938273/
      integer i,j,type
      real*8 xvalc(100) /     
     & 0.21064E+00,0.35071E+02,0.14803E+00,0.41592E+02,0.79183E+00,
     & 0.12483E+01,0.29829E+01,0.14684E+01,0.65475E+00,-.30873E+01,
     & 0.26363E+00,0.32551E+01,0.17957E+01,0.30085E+00,0.82300E-01,
     & -.44261E+00,0.15263E+00,0.20000E+00,0.39869E+01,0.32112E-01,
     & 0.00000E+00,0.12141E+00,0.21000E+00,0.00000E+00,0.21853E+00,
     & 0.11322E+00,0.50393E+02,0.29968E-02,-.29396E-03,-.69299E-01,
     & 0.18404E+02,0.13112E+00,0.17704E+00,0.10000E+01,0.14992E-01,
     & 0.55300E+02,0.52551E+00,-.70430E+00,0.70100E+02,0.20000E+01,
     & 0.10000E+01,0.95335E+00,0.10633E+01,0.98569E+00,0.92845E+00,
     & 0.98972E+00,0.95573E+00,0.10169E+01,0.97241E+00,0.98101E+00,
     & 0.10372E+01,0.98872E+00,0.10000E+01,0.10350E+01,0.10114E+01,
     & 0.95254E+00,0.11600E+01,0.12300E+01,0.11788E+01,0.10000E+01,
     & 0.99638E+00,0.11500E+01,0.99173E+00,0.10768E+01,0.10180E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01 /

      
      A = 12.0
      Z = 6.0

      read(5,*) q2
      
      
      dnu = 0.00015
      nu = 0.0
      
      do i=1,3000
        nu = nu + dnu

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

        if(ex.LE.0.012) then  !!! Only needed for plotting purposes
           RTNS = RTNS/6.0
           RLNS = RLNS/6.0
        endif
        
        RLTOT = RLTOT+RLNS
        RTTOT = RTTOT+RTNS
        
        if(q2.GT.0.0) 
     &       write(6,2000) qv,q2,ex,nu,RTTOT,RLTOT,RTQE,RLQE,RTIE,RLIE,
     &                      RTE,RLE,RTNS,RLNS          
 
        
      enddo

 2000  format(4f9.5,10E11.3)
      

      return
      end


      
      
      
      
     
CCC-----------------

      
