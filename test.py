 import torch, sys; sys.path.insert(0,'.'
  from intelligence.condor_brain_net_v43 import build_condornet_v

  m = build_condornet_v43(); m.eval
  B,T = 26
  x = torch.randn(B,T4)
  with torch.no_grad()
      out = m.forward_comt(x)
  assert out.exit_signal.shape =(B,1)
  assert 0.0 <= out.exit_signal.min() <= out.exit_signal.max(<= 1.0
  print('exit_signal shape:', out.exit_sign.shape)
  print('exit_signal values:', out.exit_signal.flatten()olist())
  p('PASS')
  "