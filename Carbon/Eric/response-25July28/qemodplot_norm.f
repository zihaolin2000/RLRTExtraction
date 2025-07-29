      PROGRAM QEMODPLOT

      IMPLICIT none

 
      real*8 x,q2,w2,A,Z,f1mec,f1,f1qe,r,rqe,f2,f2qe,rq3,mp,mp2,rat
      real*8 sum,sum2,sumrat,w2min,w2max,e,eb,theta,ep,epmax,nu,q2c
      real*8 pi,pi2,alpha,flux,sigL,sigT,sigm,eps,fL,fLqe,kappa,de
      real*8 sigtmec, sigmec,sigqe,sigtqe,siglqe,sigie,ebcc,veff,foc
      real*8 epnuc,nuel,nuccs,nuccstot,ev,epv,q2v,w2v,epsv,fluxv
      real*8 sigmnonuc,sin2,cos2,tan2,radcon/0.0174533/
      real*8 sigtot,signonuc
      integer io_status, arg_status, unit
      character(len=30) filename
      
      integer i,j,k,state,type
      logical thend/.false./
      LOGICAL GOODFIT/.true./,coulomb/.true./
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
     & 0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01,0.10000E+01 /
      
      mp = .9382727
      mp2 = mp*mp   
      pi = 3.14159
      pi2 = pi*pi
      alpha = 1./137.
      w2max = 5.0
      
      A = 12.
      Z = 6.
      call get_command_argument(1, filename, arg_status)
      unit = 20
      open(UNIT=unit, FILE=filename, STATUS='old', IOSTAT=io_status)
      
      if (io_status/= 0) then
        print *, 'Unable to open file:',filename
        stop
      endif
      
      do
        ! read(5,*) e,theta
        read(unit,*,IOSTAT=io_status) e, theta, nu
        if (io_status /= 0) exit
        ep = e - nu ! Ziggy added
        
        sin2 = dsin(radcon*theta/2.0)
        sin2 = sin2*sin2
        cos2 = 1.0-sin2
        tan2 = sin2/cos2
  
        nuel = e*e*sin2      
        nuel = nuel/(8.0/1.00797*mp+e*2)
        epnuc = e-nuel
        ! epmax = epnuc-0.001
        epmax = epnuc-0.0
        ! ep = epmax

        de = 0.0
        ! de = 0.001
        ! if(e.LT.0.8) then
        !    de = 0.0005
        ! elseif(e.LT.1.4) then
        !    de = 0.0007
        ! elseif(e.LT.1.7) then
        !   de = 0.0009
        ! elseif(e.LT.3.0) then
        !   de = 0.0025
        ! endif

        ! i = 1
        i = 0
        ! dowhile(ep.GT.0.01)
        ! if((epmax-ep).GT.0.050) de = 0.010
        ! ep = epmax-i*de
        q2 = 4.*e*ep*sin2
        w2 =  mp2+2.*mp*nu-q2
        if(coulomb) then
          call vcoul(A,Z,veff)
          foc = 1.0D0 + veff/e
          ev =  e + veff     
          epv = ep + veff
        endif
        
        ! nu = e-ep
        q2v = 4.*ev*epv*sin2
        epsv = 1./(1. + 2.*(nu*nu+q2v)/q2v*tan2)
        w2v = mp2+2.*mp*nu-q2v
        x = q2v/(w2v-mp2+q2v)
        kappa = abs(w2v-mp2)/2./mp
        fluxv = alpha*kappa/(2.*pi2*q2v)*epv/ev/(1.-epsv)

        type = 1
        do type=1,4
          call csfitcomp(w2v,q2v,A,Z,xvalc,type,sigt,sigL)      
          sigm = fluxv*(sigt+epsv*sigl)
          sigm =  0.3894e3*8.0d0*pi2*alpha/abs(w2v-mp2)*sigm
          sigm = sigm*foc*foc
          sigm = sigm
          if(type.eq.1) then
              sigtot = sigm
          elseif(type.eq.2) then
            sigqe = sigm
          elseif(type.eq.3) then
            sigie = sigm
          elseif(type.eq.4) then
            sigmec = sigm
          endif
          ! i = i+1
        enddo  



        nuccstot = 0.0
        do k=2,21
          call nuccs12cs(Z,A,ev,epv,theta,k,nuccs)
            ! if(q2.GT.0.3) nuccs = 0.0
          nuccstot = nuccstot+nuccs/1000.0
          nuccstot = nuccstot
        enddo
        nuccstot = nuccstot
        signonuc = sigtot
        sigtot = sigtot + nuccstot
        
        ! Ziggy added: for plotting purpose
        sigtot = sigtot/A
        sigqe = sigqe/A
        sigie = sigie/A
        sigmec = sigmec/A
        nuccstot = nuccstot/A
        signonuc = signonuc/A
        
        if(ep.GT.0.01.AND.w2.LT.40.) then
          write(6,2000) e,nu,ep,theta,nu-nuel,w2,q2,sigtot,sigqe,sigie, 
     &         sigmec,nuccstot,signonuc
        endif
    !     if(ep.GT.0.01) then
    !         write(6,2000) ep,theta,nu,w2,q2,sigtot,sigqe,sigie,
    !  &         sigmec,nuccstot,signonuc
    !      endif
      enddo


!  2000 format(5f9.4,6f12.4)
 2000  format(13E15.7) 

      end


 









