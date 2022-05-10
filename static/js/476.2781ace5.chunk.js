"use strict";(self.webpackChunkgenshin_optimizer=self.webpackChunkgenshin_optimizer||[]).push([[476],{96106:function(e,n,t){var i=t(4942),a=t(1413),r=t(45987),s=t(66934),o=t(18455),l=t(69293),c=t(80184),d=["className"],u=(0,s.ZP)((function(e){var n=e.className,t=(0,r.Z)(e,d);return(0,c.jsx)(o.Z,(0,a.Z)((0,a.Z)({},t),{},{arrow:!0,classes:{popper:n}}))}))((function(e){var n,t=e.theme;return n={},(0,i.Z)(n,"& .".concat(l.Z.arrow),{color:t.palette.common.black}),(0,i.Z)(n,"& .".concat(l.Z.tooltip),{backgroundColor:t.palette.common.black,maxWidth:500}),n}));n.Z=u},10600:function(e,n,t){t.d(n,{CC:function(){return h},ZP:function(){return v}});var i=t(29439),a=t(1413),r=t(45987),s=t(66934),o=t(4834),l=t(24518),c=t(72791),d=t(80184),u=["children","disableRipple","disableFocusRipple","disableTouchRipple"],p=["value","onChange","disabled","float"],x=(0,s.ZP)(o.ZP)((function(e){var n=e.theme;return{backgroundColor:n.palette.primary.main,transition:"all 0.5s ease","&:hover":{backgroundColor:n.palette.primary.dark},"&.Mui-focused":{backgroundColor:n.palette.primary.dark},"&.Mui-disabled":{backgroundColor:n.palette.primary.dark}}})),f=(0,s.ZP)(l.Z)((function(e){return{backgroundColor:e.theme.palette.primary.main,padding:0,overflow:"hidden",div:{width:"100%",height:"100%"}}}));function h(e){var n=e.children,t=(e.disableRipple,e.disableFocusRipple,e.disableTouchRipple,(0,r.Z)(e,u));return(0,d.jsx)(f,(0,a.Z)((0,a.Z)({disableRipple:!0,disableFocusRipple:!0,disableTouchRipple:!0},t),{},{children:n}))}function v(e){var n=e.value,t=void 0===n?0:n,s=e.onChange,o=e.disabled,l=void 0!==o&&o,u=e.float,f=void 0!==u&&u,h=(0,r.Z)(e,p),v=(0,c.useState)(t),m=(0,i.Z)(v,2),Z=m[0],j=m[1],b=(0,c.useState)(!1),g=(0,i.Z)(b,2),y=g[0],C=g[1],k=(0,c.useMemo)((function(){return f?parseFloat:parseInt}),[f]),w=(0,c.useCallback)((function(){s(Z),C(!1)}),[s,Z,C]),R=(0,c.useCallback)((function(){C(!0)}),[C]);(0,c.useEffect)((function(){return j(t)}),[t,j]);var P=(0,c.useCallback)((function(e){return j(k(e.target.value)||0)}),[j,k]),D=(0,c.useCallback)((function(e){return"Enter"===e.key&&w()}),[w]);return(0,d.jsx)(x,(0,a.Z)({value:y&&!Z?"":Z,"aria-label":"custom-input",type:"number",inputProps:{step:f?.1:1},onChange:P,onBlur:w,onFocus:R,disabled:l,onKeyDown:D},h))}},37503:function(e,n,t){t.d(n,{X:function(){return M},Z:function(){return H}});var i=t(68870),a=t(20890),r=t(9585),s=t(94721),o=t(72791),l=t(2693),c=t(60393),d=t(3992),u=t(39504),p=t(88034),x=t(29439),f=t(72247),h=t(9912),v=t(24518),m=t(23786),Z=t(2199),j=t(33890),b=t(25617),g=t(66624),y=t(80184);function C(e){var n=e.conditional,t=e.disabled,i=void 0!==t&&t;return 1===Object.keys(n.states).length&&"path"in n?(0,y.jsx)(k,{conditional:n,disabled:i}):"path"in n?(0,y.jsx)(w,{conditional:n,disabled:i}):(0,y.jsx)(R,{conditional:n,disabled:i})}function k(e){var n=e.conditional,t=e.disabled,i=(0,o.useContext)(l.R),a=i.character,r=i.characterDispatch,s=i.data,d=(0,o.useCallback)((function(e){var t=(0,c.I8)(a.conditional);e?(0,c.SR)(t,n.path,e):(0,c.uH)(t,n.path),r({conditional:t})}),[n,a,r]),u=s.get(n.value).value,p=(0,x.Z)(Object.entries(n.states)[0],2),m=p[0],Z=D(p[1].name),j=S(n.name);return(0,y.jsxs)(v.Z,{fullWidth:!0,size:"small",sx:{borderRadius:0},color:u?"success":"primary",onClick:function(){return d(u?void 0:m)},disabled:t,startIcon:u?(0,y.jsx)(f.Z,{}):(0,y.jsx)(h.Z,{}),children:[j," ",Z]})}function w(e){var n=e.conditional,t=e.disabled,i=(0,o.useContext)(l.R),a=i.character,r=i.characterDispatch,d=i.data,u=(0,o.useCallback)((function(e){var t=(0,c.I8)(a.conditional);e?(0,c.SR)(t,n.path,e):(0,c.uH)(t,n.path),r({conditional:t})}),[n,a,r]),p=d.get(n.value).value,f=p?n.states[p]:void 0,h=f?D(f.name):(0,y.jsx)(b.Z,{color:"secondary",children:"Not Active"}),v=S(n.name);return(0,y.jsxs)(j.Z,{fullWidth:!0,size:"small",sx:{borderRadius:0},color:p?"success":"primary",title:(0,y.jsxs)("span",{children:[v," ",h]}),disabled:t,children:[(0,y.jsx)(m.Z,{onClick:function(){return u()},selected:!f,disabled:!f,children:(0,y.jsx)("span",{children:"Not Active"})}),(0,y.jsx)(s.Z,{}),Object.entries(n.states).map((function(e){var n=(0,x.Z)(e,2),t=n[0],i=n[1];return(0,y.jsx)(m.Z,{onClick:function(){return u(t)},selected:p===t,disabled:p===t,children:i.name},t)}))]})}function R(e){var n=e.conditional,t=e.disabled,i=(0,o.useContext)(l.R),a=i.character,r=i.characterDispatch,s=i.data,d=(0,o.useCallback)((function(e,n){var t=(0,c.I8)(a.conditional);n?(0,c.SR)(t,e,n):(0,c.uH)(t,e),r({conditional:t})}),[a,r]);return(0,y.jsx)(Z.Z,{fullWidth:!0,orientation:"vertical",disableElevation:!0,color:"secondary",children:Object.entries(n.states).map((function(e){var n=(0,x.Z)(e,2),i=n[0],a=n[1],r=s.get(a.value).value,o=r===i;return(0,y.jsx)(v.Z,{color:o?"success":"primary",disabled:t,fullWidth:!0,onClick:function(){return d(a.path,r?void 0:i)},size:"small",startIcon:o?(0,y.jsx)(f.Z,{}):(0,y.jsx)(h.Z,{}),sx:{borderRadius:0},children:S(a.name)},i)}))})}function P(e){return"string"!==typeof e}function D(e){if(!e)return"";var n="primary",t=e;return e&&P(e)&&e.props.color&&(n=e.props.color,t=(0,y.jsx)("span",{children:e.props.children})),(0,y.jsx)(b.Z,{sx:{ml:.5},color:n,children:t})}function S(e){if(P(e)){var n=e.props.key18,t=e.props.ns,i=e.props.values;return(0,y.jsx)(g.v,{ns:t,key18:n,values:i,useBadge:!0})}return e}function I(e){var n,t=e.conditional,i=e.hideHeader,a=void 0!==i&&i,r=e.hideDesc,s=void 0!==r&&r,x=(0,o.useContext)(l.R).data;if("path"in t){var f,h=x.get(t.value).value;n=h&&(null===(f=t.states[h])||void 0===f?void 0:f.fields)}else n=Object.values(t.states).flatMap((function(e){return x.get(e.value).value?e.fields:[]}));return(0,y.jsxs)(d.Z,{children:[!(0,c.mY)(a,t)&&(0,y.jsx)(M,{header:t.header,hideDesc:s}),(0,y.jsx)(u.Z,{sx:{p:0,"&:last-child":{pb:0}},children:(0,y.jsx)(C,{conditional:t})}),n&&(0,y.jsx)(p.ZP,{fields:n})]})}var W=t(55221);function H(e){var n=e.sections,t=e.teamBuffOnly,a=e.hideDesc,r=void 0!==a&&a,s=e.hideHeader,c=void 0!==s&&s,d=(0,o.useContext)(l.R).data;if(!n.length)return null;var u=n.map((function(e,n){return e.canShow&&!d.get(e.canShow).value||t&&!e.teamBuff?null:(0,y.jsx)(L,{section:e,hideDesc:r,hideHeader:c},n)})).filter((function(e){return e}));return u.length?(0,y.jsx)(i.Z,{display:"flex",flexDirection:"column",gap:1,children:u}):null}function L(e){var n=e.section,t=e.hideDesc,i=void 0!==t&&t,a=e.hideHeader,r=void 0!==a&&a;return"fields"in n?(0,y.jsx)(N,{section:n,hideDesc:i,hideHeader:r}):"states"in n?(0,y.jsx)(I,{conditional:n,hideDesc:i,hideHeader:r}):(0,y.jsx)(E,{section:n})}function N(e){var n=e.section,t=e.hideDesc,i=e.hideHeader;return(0,y.jsxs)(d.Z,{children:[!(0,c.mY)(i,n)&&n.header&&(0,y.jsx)(M,{header:n.header,hideDesc:t,hideDivider:0===n.fields.length}),(0,y.jsx)(p.ZP,{fields:n.fields})]})}function E(e){var n=e.section,t=(0,o.useContext)(l.R).data;return(0,y.jsx)("div",{children:(0,c.mY)(n.text,t)})}function M(e){var n=e.header,t=e.hideDesc,i=e.hideDivider,d=(0,o.useContext)(l.R).data,u=n.icon,p=n.title,x=n.action;u=(0,c.mY)(u,d);var f=!t&&(0,c.mY)(n.description,d),h=t?p:(0,y.jsxs)("span",{children:[p," ",(0,y.jsx)(W.Z,{title:(0,y.jsx)(a.Z,{children:f})})]});return(0,y.jsxs)(y.Fragment,{children:[(0,y.jsx)(r.Z,{avatar:u,title:h,action:x,titleTypographyProps:{variant:"subtitle2"}}),!i&&(0,y.jsx)(s.Z,{})]})}},88034:function(e,n,t){t.d(n,{lD:function(){return P},JW:function(){return R},ZP:function(){return C}});var i=t(35893),a=t(15021),r=t(68870),s=t(20890),o=t(66934),l=t(90493),c=t(72791),d=t(2693),u=t(79406),p=t(60393),x=t(91702),f=t(1413),h=t(45987),v=t(53174),m=t(54483),Z=t(96106),j=t(80184),b=["className"],g=function(e){var n=e.className,t=(0,h.Z)(e,b);return(0,j.jsx)(Z.Z,(0,f.Z)((0,f.Z)({placement:"top"},t),{},{className:n,children:(0,j.jsx)(r.Z,{component:"span",sx:{cursor:"help"},children:(0,j.jsx)(m.G,{icon:v.sph})})}))},y=t(75545);function C(e){var n=e.fields;return(0,j.jsx)(P,{sx:{m:0},children:n.map((function(e,n){return(0,j.jsx)(k,{field:e,component:a.ZP},n)}))})}function k(e){var n=e.field,t=e.component,i=(0,c.useContext)(d.R),a=i.data,r=i.oldData;if(!(0,c.useMemo)((function(){var e,t;return null===(e=null===n||void 0===n||null===(t=n.canShow)||void 0===t?void 0:t.call(n,a))||void 0===e||e}),[n,a]))return null;if("node"in n){var s=a.get(n.node);if(s.isEmpty)return null;if(r){var o=r.get(n.node),l=o.isEmpty?0:o.value;return(0,j.jsx)(R,{node:s,oldValue:l,suffix:n.textSuffix,component:t})}return(0,j.jsx)(R,{node:s,suffix:n.textSuffix,component:t})}return(0,j.jsx)(w,{field:n,component:t})}function w(e){var n,t=e.field,i=e.component,a=(0,c.useContext)(d.R).data,o=(0,p.mY)(t.value,a),l=(0,p.mY)(t.variant,a),u=t.text&&(0,j.jsx)("span",{children:t.text}),x=t.textSuffix&&(0,j.jsx)("span",{children:t.textSuffix});return(0,j.jsxs)(r.Z,{width:"100%",sx:{display:"flex",justifyContent:"space-between",gap:1},component:i,children:[(0,j.jsxs)(s.Z,{color:"".concat(l,".main"),sx:{display:"flex",gap:1,alignItems:"center"},children:[u,x]}),(0,j.jsxs)(s.Z,{children:["number"===typeof o?null===(n=o.toFixed)||void 0===n?void 0:n.call(o,t.fixed):o,t.unit]})]})}function R(e){var n=e.node,t=e.oldValue,a=e.suffix,o=e.component;if(n.isEmpty)return null;a=a&&(0,j.jsx)("span",{children:a});var l=n.info.key&&y.Z[n.info.key],c=n.info.key?u.ZP.get(n.info.key):"",d=n.info.isTeamBuff,p=n.formula,f="";if(t){var h=n.value-t;f=(0,j.jsxs)("span",{children:[(0,u.EC)(t,n.unit),h>1e-4||h<-1e-4?(0,j.jsxs)(x.Z,{color:h>0?"success":"error",children:[" ",h>0?"+":"",(0,u.EC)(h,n.unit)]}):""]})}else f=(0,u.EC)(n.value,n.unit);var v=!!n.formula&&(0,j.jsx)(g,{title:(0,j.jsx)(s.Z,{children:p})});return(0,j.jsxs)(r.Z,{width:"100%",sx:{display:"flex",justifyContent:"space-between",gap:1},component:o,children:[(0,j.jsxs)(s.Z,{color:"".concat(n.info.variant,".main"),sx:{display:"flex",gap:1,alignItems:"center"},children:[!!d&&(0,j.jsx)(i.Z,{}),l,c,a]}),(0,j.jsxs)(s.Z,{sx:{display:"flex",gap:1,alignItems:"center"},children:[f,v]})]})}var P=(0,o.ZP)(l.Z)((function(e){var n=e.theme;return{borderRadius:n.shape.borderRadius,overflow:"hidden",margin:0,"> .MuiListItem-root:nth-of-type(even)":{backgroundColor:n.palette.contentDark.main},"> .MuiListItem-root:nth-of-type(odd)":{backgroundColor:n.palette.contentDarker.main}}}))},55221:function(e,n,t){var i=t(1413),a=t(45987),r=t(53174),s=t(54483),o=t(68870),l=t(96106),c=t(80184),d=["className"];n.Z=function(e){var n=e.className,t=(0,a.Z)(e,d);return(0,c.jsx)(l.Z,(0,i.Z)((0,i.Z)({placement:"top"},t),{},{className:n,children:(0,c.jsx)(o.Z,{component:"span",sx:{cursor:"help"},children:(0,c.jsx)(s.G,{icon:r.sqG})})}))}},2693:function(e,n,t){t.d(n,{R:function(){return i}});var i=(0,t(72791).createContext)({})},74476:function(e,n,t){t.r(n),t.d(n,{default:function(){return F}});var i=t(93433),a=t(29439),r=t(40117),s=t(62002),o=t(39504),l=t(61889),c=t(68870),d=t(20890),u=t(2199),p=t(24518),x=t(23786),f=t(94721),h=t(9585),v=t(15021),m=t(72791),Z=t(3992),j=t(71310),b=t(36944),g=t(68198),y=t(10600),C=t(37503),k=t(33890),w=t(88034),R=t(9321),P=t(10157),D=t(44361),S=t(947),I=t(2139),W=t(66218),H=t(56928),L=t(2693),N=t(26138),E=t(73036),M=t(42320),T=t(74480),B=t(60393),z=t(80184);function F(e){var n,t=e.weaponId,F=e.footer,J=void 0!==F&&F,O=e.onClose,Y=e.extraButtons,A=(0,m.useContext)(L.R).data,V=(0,m.useContext)(H.t).database,G=(0,T.Z)(t),K=null!==G&&void 0!==G?G:{},U=K.key,X=void 0===U?"":U,_=K.level,q=void 0===_?0:_,Q=K.refinement,$=void 0===Q?0:Q,ee=K.ascension,ne=void 0===ee?0:ee,te=K.lock,ie=K.location,ae=void 0===ie?"":ie,re=K.id,se=(0,M.Z)(W.Z.get(X),[X]),oe=(0,m.useCallback)((function(e){V.updateWeapon(e,t)}),[t,V]),le=(0,m.useCallback)((function(e){e=(0,B.uZ)(e,1,90);var n=I.SJ.findIndex((function(n){return e<=n}));oe({level:e,ascension:n})}),[oe]),ce=(0,m.useCallback)((function(){var e=I.SJ.findIndex((function(e){return 90!==q&&q===e}));oe(ne===e?{ascension:ne+1}:{ascension:e})}),[oe,ne,q]),de=(0,M.Z)(ae?S.Z.get(ae):void 0,[ae]),ue=de?function(e){return e.weaponType===de.weaponTypeKey}:void 0,pe=de&&de.weaponTypeKey,xe=(0,m.useCallback)((function(e){return re&&V.setWeaponLocation(re,e)}),[V,re]),fe=(0,m.useCallback)((function(e){return e.weaponTypeKey===(null===se||void 0===se?void 0:se.weaponType)}),[se]),he=(0,m.useState)(!1),ve=(0,a.Z)(he,2),me=ve[0],Ze=ve[1],je=ne<2?null===se||void 0===se?void 0:se.img:null===se||void 0===se?void 0:se.imgAwaken;(0,m.useEffect)((function(){if(se&&oe&&se.key===(null===G||void 0===G?void 0:G.key)&&se.rarity<=2&&(q>70||ne>4)){var e=(0,a.Z)(I.Xu[0],2),n=e[0],t=e[1];oe({level:n,ascension:t})}}),[se,G,oe,q,ne]);var be=(0,m.useMemo)((function(){return se&&G&&(0,E.mP)([se.data,(0,E.v0)(G)])}),[se,G]);return(0,z.jsx)(R.Z,{open:!!t,onClose:O,containerProps:{maxWidth:"md"},children:(0,z.jsxs)(j.Z,{children:[(0,z.jsx)(D.Z,{show:me,onHide:function(){return Ze(!1)},onSelect:function(e){return oe({key:e})},filter:ue,weaponFilter:pe}),(0,z.jsx)(o.Z,{children:se&&be&&(0,z.jsxs)(l.ZP,{container:!0,spacing:1.5,children:[(0,z.jsx)(l.ZP,{item:!0,xs:12,sm:3,children:(0,z.jsxs)(l.ZP,{container:!0,spacing:1.5,children:[(0,z.jsx)(l.ZP,{item:!0,xs:6,sm:12,children:(0,z.jsx)(c.Z,{component:"img",src:je,className:"grad-".concat(se.rarity,"star"),sx:{maxWidth:256,width:"100%",height:"auto",borderRadius:1}})}),(0,z.jsx)(l.ZP,{item:!0,xs:6,sm:12,children:(0,z.jsx)(d.Z,{children:(0,z.jsx)("small",{children:se.description})})})]})}),(0,z.jsxs)(l.ZP,{item:!0,xs:12,sm:9,sx:{display:"flex",flexDirection:"column",gap:1},children:[(0,z.jsx)(c.Z,{display:"flex",gap:1,flexWrap:"wrap",justifyContent:"space-between",children:(0,z.jsxs)(u.Z,{children:[(0,z.jsx)(p.Z,{onClick:function(){return Ze(!0)},children:null!==(n=null===se||void 0===se?void 0:se.name)&&void 0!==n?n:"Select a Weapon"}),(null===se||void 0===se?void 0:se.hasRefinement)&&(0,z.jsxs)(k.Z,{title:"Refinement ".concat($),children:[(0,z.jsx)(x.Z,{children:"Select Weapon Refinement"}),(0,z.jsx)(f.Z,{}),(0,i.Z)(Array(5).keys()).map((function(e){return(0,z.jsx)(x.Z,{onClick:function(){return oe({refinement:e+1})},selected:$===e+1,disabled:$===e+1,children:"Refinement ".concat(e+1)},e)}))]}),Y]})}),(0,z.jsxs)(c.Z,{display:"flex",gap:1,flexWrap:"wrap",justifyContent:"space-between",children:[(0,z.jsxs)(u.Z,{sx:{bgcolor:function(e){return e.palette.contentLight.main}},children:[(0,z.jsx)(y.CC,{children:(0,z.jsx)(y.ZP,{onChange:le,value:q,startAdornment:"Lv. ",inputProps:{min:1,max:90,sx:{textAlign:"center"}},sx:{width:"100%",height:"100%",pl:2}})}),se&&(0,z.jsx)(p.Z,{sx:{pl:1},disabled:!se.ambiguousLevel(q),onClick:ce,children:(0,z.jsxs)("strong",{children:["/ ",I.SJ[ne]]})}),se&&(0,z.jsx)(k.Z,{title:"Select Level",children:se.milestoneLevels.map((function(e){var n=(0,a.Z)(e,2),t=n[0],i=n[1],r=t===I.SJ[i]?"Lv. ".concat(t):"Lv. ".concat(t,"/").concat(I.SJ[i]),s=t===q&&i===ne;return(0,z.jsx)(x.Z,{selected:s,disabled:s,onClick:function(){return oe({level:t,ascension:i})},children:r},"".concat(t,"/").concat(i))}))})]}),(0,z.jsx)(p.Z,{color:"error",onClick:function(){return re&&V.updateWeapon({lock:!te},re)},startIcon:te?(0,z.jsx)(r.Z,{}):(0,z.jsx)(s.Z,{}),children:te?"Locked":"Unlocked"})]}),(0,z.jsx)(d.Z,{children:(0,z.jsx)(P.t,{stars:se.rarity})}),(0,z.jsx)(d.Z,{variant:"subtitle1",children:(0,z.jsx)("strong",{children:se.passiveName})}),(0,z.jsx)(d.Z,{gutterBottom:!0,children:se.passiveName&&se.passiveDescription(be.get(N.ri.weapon.refineIndex).value)}),(0,z.jsxs)(c.Z,{display:"flex",flexDirection:"column",gap:1,children:[(0,z.jsxs)(Z.Z,{children:[(0,z.jsx)(h.Z,{title:"Main Stats",titleTypographyProps:{variant:"subtitle2"}}),(0,z.jsx)(f.Z,{}),(0,z.jsx)(w.lD,{children:[N.ri.weapon.main,N.ri.weapon.sub,N.ri.weapon.sub2].map((function(e,n){var t=be.get(e);return t.isEmpty||!t.value?null:(0,z.jsx)(w.JW,{node:t,component:v.ZP},t.info.key)}))})]}),A&&se.document&&(0,z.jsx)(C.Z,{sections:se.document})]})]})]})}),J&&re&&(0,z.jsx)(o.Z,{sx:{py:1},children:(0,z.jsxs)(l.ZP,{container:!0,children:[(0,z.jsx)(l.ZP,{item:!0,flexGrow:1,children:(0,z.jsx)(b.Z,{noUnselect:!0,inventory:!0,value:ae,onChange:xe,filter:fe})}),!!O&&(0,z.jsx)(l.ZP,{item:!0,children:(0,z.jsx)(g.Z,{large:!0,onClick:O})})]})})]})})}},72247:function(e,n,t){var i=t(76189),a=t(80184);n.Z=(0,i.Z)((0,a.jsx)("path",{d:"M19 3H5c-1.11 0-2 .9-2 2v14c0 1.1.89 2 2 2h14c1.11 0 2-.9 2-2V5c0-1.1-.89-2-2-2zm-9 14-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"}),"CheckBox")},9912:function(e,n,t){var i=t(76189),a=t(80184);n.Z=(0,i.Z)((0,a.jsx)("path",{d:"M19 5v14H5V5h14m0-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2z"}),"CheckBoxOutlineBlank")}}]);
//# sourceMappingURL=476.2781ace5.chunk.js.map